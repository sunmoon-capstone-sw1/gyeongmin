import sys

from PIL import Image
import numpy as np
from attrdict import AttrDict

import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

class Triton:

    def __init__(self, model_name, img_path):
        self.MODEL_NAME = model_name
        self.SCALING = "INCEPTION"
        self.MODEL_VERSION = ""
        self.BATCH_SIZE = 1
        self.VERBOSE = False
        self.ASYNC = False
        self.CLASSES = 1
        self.URL = "localhost:8000"
        self.IMG_PATH = img_path

    def __call__(self):
        self.make_triton_client()
        self.make_model_metadate()
        self.make_model_config()
        self.convert_http_metadata_config()
        self.parse_model()
        self.get_filenames()
        self.do_preprocess()
        self.send_request()
        self.print_result()

    def parse_model(self):
    
        if len(self.model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(self.model_metadata.inputs)))
        if len(self.model_metadata.outputs) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(self.model_metadata.outputs)))

        if len(self.model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(self.model_config.input)))

        input_metadata = self.model_metadata.inputs[0]
        input_config = self.model_config.input[0]
        output_metadata = self.model_metadata.outputs[0]

        if output_metadata.datatype != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            self.model_metadata.name + "' output type is " +
                            output_metadata.datatype)

        output_batch_dim = (self.model_config.max_batch_size > 0)
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        input_batch_dim = (self.model_config.max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, self.model_metadata.name,
                    len(input_metadata.shape)))

        if type(input_config.format) == str:
            FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
            input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

        if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
            (input_config.format != mc.ModelInput.FORMAT_NHWC)):
            raise Exception("unexpected input format " +
                            mc.ModelInput.Format.Name(input_config.format) +
                            ", expecting " +
                            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                            " or " +
                            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]

        self.max_batch_size = self.model_config.max_batch_size
        self.input_name = input_metadata.name
        self.output_name = output_metadata.name
        self.c = c
        self.h = h
        self.w = w
        self.format = input_config.format
        self.dtype = input_metadata.datatype

    def preprocess(self, img, dtype, h, w):

        sample_img = img.convert('RGB')

        resized_img = sample_img.resize((w, h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        npdtype = triton_to_np_dtype(dtype)
        typed = resized.astype(npdtype)

        scaled = (typed / 127.5) - 1

        ordered = scaled # Channels are in RGB order.

        return ordered

    def postprocess(self, results, output_name, batch_size, batching):

        output_array = results.as_numpy(output_name)
        if len(output_array) != batch_size:
            raise Exception("expected {} results, got {}".format(
                batch_size, len(output_array)))

        # Include special handling for non-batching models
        for results in output_array:
            if not batching:
                results = [results]
            for result in results:
                if output_array.dtype.type == np.object_:
                    cls = "".join(chr(x) for x in result).split(':')
                else:
                    cls = result.split(':')
                print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))
                print(">>> Result of prediction is", cls[2])

    def requestGenerator(self, batched_image_data, input_name, output_name, dtype):

        client = httpclient

        # Set the input data
        self.inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
        self.inputs[0].set_data_from_numpy(batched_image_data)

        self.outputs = [
            client.InferRequestedOutput(output_name, class_count=self.CLASSES)
        ]

        yield self.inputs, self.outputs, self.MODEL_NAME, self.MODEL_VERSION
    
    def make_triton_client(self):
        try:
            self.triton_client = httpclient.InferenceServerClient(
                url=self.URL, verbose=self.VERBOSE, concurrency=1)
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

    def make_model_metadate(self):
        try:
            self.model_metadata = self.triton_client.get_model_metadata(
                model_name=self.MODEL_NAME, model_version=self.MODEL_VERSION)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)
    
    def make_model_config(self):
        try:
            self.model_config = self.triton_client.get_model_config(
                model_name=self.MODEL_NAME, model_version=self.MODEL_VERSION)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)
    
    def convert_http_metadata_config(self):
        self.model_metadata = AttrDict(self.model_metadata)
        self.model_config = AttrDict(self.model_config)

    def get_filenames(self):
        self.filenames = [self.IMG_PATH, ]
        self.filenames.sort()
    
    def do_preprocess(self):
        self.image_data = []
        for filename in self.filenames:
            img = Image.open(filename)
            self.image_data.append(
                self.preprocess(img, self.dtype, self.h, self.w))
    
    def send_request(self):
        self.requests = []
        self.responses = []
        self.result_filenames = []
        self.request_ids = []
        self.image_idx = 0
        self.last_request = False
        self.sent_count = 0

        while not self.last_request:
            self.input_filenames = []
            self.repeated_image_data = []

            for idx in range(self.BATCH_SIZE):
                self.input_filenames.append(self.filenames[self.image_idx])
                self.repeated_image_data.append(self.image_data[self.image_idx])
                self.image_idx = (self.image_idx + 1) % len(self.image_data)
                if self.image_idx == 0:
                    self.last_request = True

            batched_image_data = self.repeated_image_data[0]

            # Send request
            try:
                for inputs, outputs, model_name, model_version in self.requestGenerator(
                        batched_image_data, self.input_name, self.output_name, self.dtype):
                    self.sent_count += 1
                    self.responses.append(
                        self.triton_client.infer(model_name,
                                            inputs,
                                            request_id=str(self.sent_count),
                                            model_version=model_version,
                                            outputs=outputs))

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                sys.exit(1)

    def print_result(self):
        for response in self.responses:
            this_id = response.get_response()["id"]
            print("RESPONSE".center(40, "="))
            print("Request {}, batch size {}".format(this_id, self.BATCH_SIZE))
            self.postprocess(response, self.output_name, self.BATCH_SIZE, self.max_batch_size > 0)
            print("".center(40, "="))

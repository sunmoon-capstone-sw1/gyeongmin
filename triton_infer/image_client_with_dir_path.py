import os
from re import VERBOSE
import sys

from PIL import Image
import numpy as np
from attrdict import AttrDict

import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
import queue

MODEL_NAME = "mobilenet_money_detection"
SCALING = "INCEPTION"
MODEL_VERSION = ""
BATCH_SIZE = 1
VERBOSE = False
ASYNC = False
CLASSES = 1
URL = "localhost:8000"
DIR_PATH = "C:/Users/2019A00298/test_img_before_make_model"
# DIR_PATH = ""
IMG_PATH = "C:/Users/2019A00298/test_img_before_make_model/dollor_10.jpg"

class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()

def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
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

    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, c, h, w, input_config.format,
            input_metadata.datatype)


def preprocess(img, dtype, h, w, scaling):

    sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]


    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1

    ordered = scaled # Channels are in RGB order.

    return ordered


def postprocess(results, output_name, batch_size, batching):

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


def requestGenerator(batched_image_data, input_name, output_name, dtype):

    client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(output_name, class_count=CLASSES)
    ]

    yield inputs, outputs, MODEL_NAME, MODEL_VERSION


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


try:
    triton_client = httpclient.InferenceServerClient(
        url=URL, verbose=VERBOSE, concurrency=1)
except Exception as e:
    print("client creation failed: " + str(e))
    sys.exit(1)

# Make sure the model matches our requirements, and get some
# properties of the model that we need for preprocessing
try:
    model_metadata = triton_client.get_model_metadata(
        model_name=MODEL_NAME, model_version=MODEL_VERSION)
except InferenceServerException as e:
    print("failed to retrieve the metadata: " + str(e))
    sys.exit(1)

try:
    model_config = triton_client.get_model_config(
        model_name=MODEL_NAME, model_version=MODEL_VERSION)
except InferenceServerException as e:
    print("failed to retrieve the config: " + str(e))
    sys.exit(1)


model_metadata, model_config = convert_http_metadata_config(
    model_metadata, model_config)

max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
    model_metadata, model_config)

filenames = []
if os.path.isdir(DIR_PATH):
    filenames = [
        os.path.join(DIR_PATH, f)
        for f in os.listdir(DIR_PATH)
        if os.path.isfile(os.path.join(DIR_PATH, f))
    ]
else:
    filenames = [
        IMG_PATH,
    ]

filenames.sort()

# Preprocess the images into input data according to model
# requirements
image_data = []
for filename in filenames:
    img = Image.open(filename)
    image_data.append(
        preprocess(img, dtype, h, w, SCALING))

# Send requests of BATCH_SIZE images. If the number of
# images isn't an exact multiple of BATCH_SIZE then just
# start over with the first images until the batch is filled.
requests = []
responses = []
result_filenames = []
request_ids = []
image_idx = 0
last_request = False
user_data = UserData()

sent_count = 0

while not last_request:
    input_filenames = []
    repeated_image_data = []

    for idx in range(BATCH_SIZE):
        input_filenames.append(filenames[image_idx])
        repeated_image_data.append(image_data[image_idx])
        image_idx = (image_idx + 1) % len(image_data)
        if image_idx == 0:
            last_request = True

    batched_image_data = repeated_image_data[0]

    # Send request
    try:
        for inputs, outputs, model_name, model_version in requestGenerator(
                batched_image_data, input_name, output_name, dtype):
            sent_count += 1
            responses.append(
                triton_client.infer(model_name,
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=model_version,
                                    outputs=outputs))

    except InferenceServerException as e:
        print("inference failed: " + str(e))
        sys.exit(1)


for response in responses:
    this_id = response.get_response()["id"]
    print("RESPONSE".center(40, "="))
    print("Request {}, batch size {}".format(this_id, BATCH_SIZE))
    postprocess(response, output_name, BATCH_SIZE, max_batch_size > 0)

print("".center(40, "="))
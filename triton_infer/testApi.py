import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
from PIL import Image
import numpy as np
from attrdict import AttrDict
import cv2
import json
from IPython.display import display
from matplotlib import pyplot as plt

tr_model = 'yolo_money_detection'

headers = {
    "content-type": "application/json",
    "model-deployment-sn": "14",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxNEBjaGVldGFoLW1vZGVsLWRlcGxveW1lbnQiLCJuYmYiOjE2MzcxMzYzNzMsImlzcyI6IjE0QGNoZWV0YWgtbW9kZWwtZGVwbG95bWVudCIsImV4cCI6MTYzNzEzOTk3MywiaWF0IjoxNjM3MTM2MzczLCJtZHMiOjE0LCJqdGkiOiJhNzU2MDM5Mi1mOGY0LTQ5NmYtOGNhYS0yZDMxMDViZjBiM2QifQ.idLLZvU-Q9zggVpOhqie3AzbIDdMHuGxvSBTOGL9YQA"
}


def preprocess(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img.transpose((2, 0, 1)).astype(np.float32)
    img = img.transpose((1, 0, 2)).astype(np.float32)

    img /= 255.0
    return img


# Create server context
try:
    triton_client = httpclient.InferenceServerClient(
        url="localhost:8000",
        verbose=None)
except Exception as e:
    print("context creation failed: " + str(e))

# Health check
if not triton_client.is_server_live(headers=headers):
    print("FAILED : is_server_live")
    exit

if not triton_client.is_server_ready(headers=headers):
    print("FAILED : is_server_ready")
    exit

if not triton_client.is_model_ready(tr_model, headers=headers):
    print("FAILED : is_model_ready")
    exit

print('Health check OK')

# Model metadata
try:
    model_metadata = triton_client.get_model_metadata(tr_model, headers=headers)
    print('------------------------------------------------------------------')
    print('- model metadata')
    print('------------------------------------------------------------------')
    print(json.dumps(model_metadata, indent=2))
    print('------------------------------------------------------------------')

except InferenceServerException as ex:
    if "Request for unknown model" not in ex.message():
        print("FAILED : get_model_metadata")
        print("Got: {}".format(ex.message()))
        exit
    else:
        print("FAILED : get_model_metadata")
        exit

# Model configuration
try:
    model_config = triton_client.get_model_config(tr_model, headers=headers)

    if not (model_config.get('name') == tr_model):
        print("FAILED: get_model_config")
        exit

    print('- model config')
    print('------------------------------------------------------------------')
    print(json.dumps(model_config, indent=2))
    print('------------------------------------------------------------------')
except InferenceServerException as ex:
    print("FAILED : get_model_config")
    print("Got: {}".format(ex.message()))
    exit

image_data = []

width=416
height=416

inputs = []
outputs = []
inputs.append(httpclient.InferInput('input_5', [1, width, height, 3], "FP32"))


print("Creating buffer from image file...")
input_image = cv2.imread("yolo_img.png")

print("origin img shape", input_image.shape) #(1280, 960, 3)

input_image_buffer = preprocess(input_image, [width, height])

print("input_image_bugger.shape: ", input_image_buffer.shape)

input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

print("input_image_bugger.shape: ", input_image_buffer.shape)

inputs[0].set_data_from_numpy(input_image_buffer)

print("Invoking inference...")

results = triton_client.infer(model_name=tr_model,
                             inputs=inputs,
                             outputs=outputs,
                             headers=headers)

# print(results.get_output('Softmax'))
response = results.get_response()

print(f"out1_data name: {response.get('outputs')[0].get('name')}, name: {response.get('outputs')[0].get('datatype')}, name: {response.get('outputs')[0].get('shape')}")
print(f"out2_data name: {response.get('outputs')[1].get('name')}, name: {response.get('outputs')[0].get('datatype')}, name: {response.get('outputs')[0].get('shape')}")
# print(f"name: {response.get('outputs')[0].get('data')}")
out1_data = response.get('outputs')[0].get('data')
out2_data = response.get('outputs')[1].get('data')
print(f"out1_data type:{type(out1_data)}, length:{len(out1_data)}")
print(f"out2_data type:{type(out2_data)}, length:{len(out2_data)}")

print("sample 10 of output1:")
print(out1_data[:10])
print("===")
print("sample 10 of output2:")
print(out2_data[:10])
print("Done")


statistics = triton_client.get_inference_statistics(model_name=tr_model, headers=headers)

if len(statistics.get('model_stats')) != 1:
    print("FAILED: get_inference_statistics")
    exit

print(json.dumps(statistics, indent=2))

print("Done")
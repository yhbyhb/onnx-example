import cv2
import onnxruntime as ort
import numpy as np
import time

print(ort.__version__)

input_filename = "sample.jpg"
output_filename = "sample_upscaled.jpg"
model_name ='realesr-general-x4v3'

half = True
if half:
    model_filename = model_name + '-fp16' + '.onnx'
else:
    model_filename = model_name + '.onnx'

img = cv2.imread(input_filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_data_normalized = img.astype(np.float32) / 255.0

provider = "DmlExecutionProvider"

if not provider in ort.get_available_providers():
    print(f"provider {provider} not found in {ort.get_available_providers()}")
    exit()

providers = [provider]

sess_options = ort.SessionOptions()
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(model_filename, sess_options=sess_options, providers=providers, provider_options=[{'device_id': 1}])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# measure time onnx
for i in range(10):
    input_data_blob = input_data_normalized.transpose(2, 0, 1)[np.newaxis, ...]
    if half:
        input_data_blob = input_data_blob.astype(np.float16)
    start_time = time.time()
    output_data_blob = sess.run([output_name], {input_name: input_data_blob})[0]
    end_time = time.time()
    output_img = output_data_blob.squeeze().transpose(1, 2, 0)
    print("onnx inference time: ", end_time - start_time)

output_img = (output_img * 255).round().clip(0, 255).astype(np.uint8)
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_filename, output_img)

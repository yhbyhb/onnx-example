import cv2
import onnxruntime as ort
import numpy as np
import time

print(ort.get_available_providers())
print(ort.__version__)

img = cv2.imread("sample.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_data_normalized = img.astype(np.float32) / 255.0

sess = ort.InferenceSession('realesr-general-x4v3.onnx', providers=['DmlExecutionProvider'], provider_options=[{'device_id': 1}])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# measure time 
for i in range(10):
    start_time = time.time()
    input_data_blob = input_data_normalized.transpose(2, 0, 1)[np.newaxis, ...]
    output_data_blob = sess.run([output_name], {input_name: input_data_blob})[0]
    output_img = output_data_blob.squeeze().transpose(1, 2, 0)
    end_time = time.time()
    print("upscale time: ", end_time - start_time)

output_img = (output_img * 255).round().clip(0, 255).astype(np.uint8)
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("sample_upscaled.jpg", output_img)

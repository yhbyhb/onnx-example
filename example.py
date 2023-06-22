import cv2
import onnxruntime as ort
import numpy as np
import time

print(ort.get_available_providers())
print(ort.__version__)

img = cv2.imread("sample.jpg")
input_data = img / 255.0
input_data = input_data.transpose(2, 0, 1)
input_data = input_data[np.newaxis, ...].astype(np.float32)
print(input_data.shape)

sess = ort.InferenceSession('realesr-general-x4v3.onnx', providers=['DmlExecutionProvider'], provider_options=[{'device_id': 1}])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# measure time 
start_time = time.time()
output = sess.run([output_name], {input_name: input_data})[0]
end_time = time.time()
print("elapsed time: ", end_time - start_time)

print(type(output))
print(output.shape)

output_img = output.squeeze().clip(0, 1).transpose(1, 2, 0)
output_img = (output_img * 255).round().astype(np.uint8)
cv2.imwrite("sample_upscaled.jpg", output_img)
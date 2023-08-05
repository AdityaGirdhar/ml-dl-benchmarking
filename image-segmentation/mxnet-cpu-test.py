from PIL import Image
import numpy as np
import mxnet as mx
import pandas as pd
import time
import matplotlib.pyplot as plt
import sys


data_ctx = mx.cpu()
model_ctx = mx.cpu()

img = Image.open('peppers.png').resize((200, 150))
# print(img)
# converting the image to numpy array
image = np.array(img)

# converting the numpy image to mxnet array
mx_image = mx.nd.array(image).copyto(data_ctx)
mx_image = mx_image.astype('float32')/255

# getting the dimensions of the image and setting hyperparameters
width = mx_image.shape[0]
height = mx_image.shape[1]
colors = mx_image.shape[2]
num_of_epochs = 10
bandwidth = 0.1

# print(mx_image.shape)
# plt.figure()
# plt.imshow(mx_image.asnumpy())
# plt.show()

start = time.time()

# training 
for epoch in range(num_of_epochs):
    mx_image2 = mx_image.copyto(data_ctx)
    for w in range(width):
        for h in range(height):
            k = mx.nd.exp(mx.nd.sum(-mx.nd.square((mx_image[w,h] - mx_image)/bandwidth), axis = 2))
            a = mx.nd.sum((mx_image-mx_image[w,h])*k.reshape(width, height, 1), axis=(0,1))
            b = mx.nd.sum(k, axis=(0,1))
            mx_image2[w,h] = mx_image[w,h] + a/b
    mx_image = mx_image2.copyto(data_ctx)
end = time.time()

# plt.show()

with open('time.txt', 'a') as f:
    # Redirect stdout to the file
    sys.stdout = f
    print(f"MXNet (CPU): {end-start} seconds")

# # %%
# print(mx_image.shape)
# plt.figure()
# plt.imshow(mx_image.asnumpy())
# plt.show()


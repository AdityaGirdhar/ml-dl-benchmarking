# mean shift in keras

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
import time

def mean_shift(img, h=0.1, n=10):      # mean_shift algorithm
    Hgt, Wth = img.shape[:2]    # get the height and width of the image
    img1 = img.copy()
    img2 = img.copy()
    for i in range(n):  # for all iterations
        for y in range(Hgt):    # for height
            for x in range(Wth):    # for width, thus we now operate on a pixel (square)
                img_tensor = keras.backend.expand_dims(img1[y, x], 0)
                k = keras.backend.exp(-keras.backend.sum(keras.backend.square((img1[y, x]-img1)/h), axis=-1))
                a = keras.backend.sum((img1 - img_tensor) * keras.backend.expand_dims(k, -1), axis=(0, 1))
                g = a / keras.backend.sum(k)
                img2[y, x] += g
        img1 = img2
        print(f"Iteration {i+1} done\n")
    return img1

img = Image.open('peppers.png').resize((200, 150))
img = np.array(img)/255   #convert image to an array

plt.figure()
plt.imshow(img)
plt.show()

start_time = time.time()
img_s = mean_shift(img.astype(np.float32), h=0.1, n=10) # here h is the bandwidth and n is the number of iterations for which we get the clusters and update the clusters
end_time = time.time() 

plt.figure()
plt.imshow(img_s)
plt.savefig('meanshift_changed.png')
plt.show()

print("Mean shift execution time:", end_time - start_time, "seconds")
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys

def mean_shift(img, h=0.1, n=10):      # mean_shift algorithm
    Hgt, Wth = img.shape[:2]    # get the height and width of the image
    img1 = img.copy()
    img2 = img.copy()
    for i in range(n):  # for all iterations
        for y in range(Hgt):    # for height
            for x in range(Wth):    # for width, thus we now operate on a pixel (square)
                k = np.exp(-np.square((img1[y, x]-img1)/h).sum(-1))
                a = ((img1-img1[y, x]) * np.expand_dims(k,-1)).sum((0,1))
                g = a/k.sum()
                img2[y, x] += g
        img1 = img2
    return img1

img = Image.open('peppers.png').resize((200, 150))
img = np.array(img)/255   #convert image to an array

# plt.figure()
# plt.imshow(img)
# plt.show()

start = time.time()
img_s = mean_shift(img, h=0.1, n=10)    # here h is the bandwidth and n is the number of iterations
end = time.time()

with open('time.txt', 'a') as f:
    # Redirect stdout to the file
    sys.stdout = f
    print(f"Numpy: {end-start} seconds")

# plt.figure()
# plt.imshow(img_s)
# plt.savefig('Q3_changed.png')
# plt.show()



import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import sys
import time

device = torch.device("cpu")

def mean_shift(img, h=0.1, n=10):
    hgt, wth = img.shape[:2]
    img1 = img.clone().to(device)
    img2 = img.clone().to(device)
    for i in range(n):
        for y in range(hgt):
            for x in range(wth):
                k = torch.exp(-torch.square((img1[y, x] - img1) / h).sum(-1)).to(device)
                a = ((img1 - img1[y, x]) * torch.unsqueeze(k, -1)).sum((0, 1)).to(device)
                g = a/k.sum()
                img2[y, x] += g
        img1 = img2
    return img1

img = Image.open('peppers.png').resize((200, 150))
img = torch.from_numpy(np.array(img)/255).to(device) # convert to PyTorch tensor

start = time.time()
img_s = mean_shift(img, h=0.1, n=10)
end = time.time() 

with open('time.txt', 'a') as f:
    # Redirect stdout to the file
    sys.stdout = f
    print(f"PyTorch: {end-start} seconds")
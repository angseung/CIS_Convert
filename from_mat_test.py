import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from CIS_Processing import *
from CIS_Utils import *

orig_image = loadmat("testimg.mat")["noise_img"]
noised_img = np.zeros(orig_image.shape, dtype = np.uint8)
denoised_img = np.zeros(orig_image.shape, dtype = np.uint8)
#
# for i in range(orig_image.shape[2]):
#     noised_img[:, :, i] = salt_and_pepper(orig_image[:, :, i], 0.02)
#
for i in range(orig_image.shape[2]):
    denoised_img[:, :, i] = hw_RSEPD(orig_image[:, :, i], 20)

fig = plt.figure(0, figsize = [10, 20])
plt.subplot(3,1,1)
plt.imshow(orig_image)
plt.subplot(3,1,2)
plt.imshow(noised_img)
plt.subplot(3,1,3)
plt.imshow(denoised_img)

plt.show()
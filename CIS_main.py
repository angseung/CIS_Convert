import numpy as np
from CIS_Processing import *
from CIS_Utils import *
import cv2
from matplotlib import pyplot as plt

testImages = ["testimage.jpg"]
numImages = len(testImages)

noiseRatio = [0.02]
numNoise = len(noiseRatio)

# filters = ["median","HW-SEPD", "HW-DTBDM", "HW-RSEPD"]
filters = ["HW-RSEPD"]
numFilters = len(filters)

psnrs = np.zeros((numFilters, numNoise))

if (numFilters >= 7):
    raise NotImplementedError("Too many filters are considered!!!")
    plotshape = 0

else:
    plotshape = [13,23,23,23,33,33,33]

for nr in range(numNoise):
    for imnum in range(numImages):
        ori_image = imgload(testImages[imnum])
        image = add_salt_pepper_noise([ori_image])
        plt.imshow(image[0])
        plt.show()



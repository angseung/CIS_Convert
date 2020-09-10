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
        fig = plt.figure(1)
        plt.subplot(4,1,1)
        plt.imshow(ori_image)

        image_noise = add_salt_pepper_noise([ori_image], 0.01)
        plt.subplot(4,1,2)
        plt.imshow(image_noise[0])
        # plt.show()
        # fig.savefig("Salt_and_Pepper_Noised_IMG.png")

        image_NR = hw_RSEPD(image_noise[0], 20)
        image_JRT = paper_jrt(image_noise[0], 4)





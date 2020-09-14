import numpy as np
from CIS_Processing import *
from CIS_Utils import *
import cv2
from matplotlib import pyplot as plt
import os
from scipy.io import loadmat

# testImages = ["testimage.jpg"]
# testImages = ["67_8D5U5597.tiff"]
testImages = os.listdir("input_images")
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
        # if ((testImages[imnum] == ".DS_Store") or (testImages[imnum] == "Gray Scale")):
        if (not ("jpg" in testImages[imnum])):
            continue

        ori_image = imgload(testImages[imnum], 'RGB')
        print("Processing %s File..." %testImages[imnum])
        fig = plt.figure(1, figsize = [10, 20])
        plt.subplot(4,1,1)
        plt.title("Original Image")
        plt.imshow(ori_image)

        # image_noise = add_salt_pepper_noise([ori_image], 0.0001, 'RGB')
        image_noise = salt_and_pepper(ori_image, 0.01)
        plt.subplot(4,1,2)
        plt.title("Noised Image")
        # plt.imshow(image_noise[0])
        plt.imshow(image_noise)
        # plt.show()
        # fig.savefig("Salt_and_Pepper_Noised_IMG.png")

        image_NR = np.zeros(ori_image.shape, dtype = np.uint8)
        # image_NR[:, :, 0] = hw_RSEPD(image_noise[0][:, :, 0], 20)
        # image_NR[:, :, 1] = hw_RSEPD(image_noise[0][:, :, 1], 20)
        # image_NR[:, :, 2] = hw_RSEPD(image_noise[0][:, :, 2], 20)
        image_NR[:, :, 0] = hw_RSEPD(image_noise[:, :, 0], 20)
        image_NR[:, :, 1] = hw_RSEPD(image_noise[:, :, 1], 20)
        image_NR[:, :, 2] = hw_RSEPD(image_noise[:, :, 2], 20)

        plt.subplot(4,1,3)
        plt.title("Noise Reduced Image")
        plt.imshow(image_NR)


        # image_JRT = paper_jrt(image_noise[0], 4)
        # plt.subplot(4,1,4)
        # plt.title("JRT applied Image")
        # plt.imshow(image_JRT)

        plt.show()
        fig.savefig("Test_Results/Processed_IMG_%s.png" %testImages[imnum])






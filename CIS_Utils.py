import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from random import random

def imgload(name = "", mode = 'RGB'):
    img = Image.open("input_images/" + name)

    # if (img.mode is not 'RGB'):
    #     raise NotImplementedError()

    im = np.array(img)


    return im

def add_salt_pepper_noise(X_imgs = None, amount = 0.01, mode = 'RGB'):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    if (mode == 'RGB'):
        row, col, ch = X_imgs_copy[0].shape

        salt_vs_pepper = 0.2
        # amount = 0.04
        num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

        for X_img in X_imgs_copy:
            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
            X_img[coords[0], coords[1], :] = 1

            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
            X_img[coords[0], coords[1], :] = 0

    elif (mode == 'L'):
        row, col = X_imgs_copy[0].shape

        salt_vs_pepper = 0.2
        # amount = 0.04
        num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

        for X_img in X_imgs_copy:
            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
            X_img[coords[0], coords[1]] = 1

            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
            X_img[coords[0], coords[1]] = 0

    # salt_vs_pepper = 0.2
    # # amount = 0.04
    # num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    # num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    #
    # for X_img in X_imgs_copy:
    #     # Add Salt noise
    #     coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
    #     X_img[coords[0], coords[1], :] = 1
    #
    #     # Add Pepper noise
    #     coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
    #     X_img[coords[0], coords[1], :] = 0

    return X_imgs_copy

def salt_and_pepper(image, p):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - p
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random()
            if rdn < p:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]

    return output

import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from random import random
import time
import cv2

ELAPSED_TIME_OPT = True

def imgload(name = "", mode = 'RGB'):
    start_time = time.time()
    img = Image.open("input_images/" + name)

    # if (img.mode is not 'RGB'):
    #     raise NotImplementedError()

    im = np.array(img)

    elapsed_time = time.time() - start_time
    if (ELAPSED_TIME_OPT):
        print("Elapsed Time of imgload : %d (sec)" %elapsed_time)

    return im

def imgload_cv(name = "", mode = 'RGB'):
    start_time = time.time()

    if (mode == "RGB"):
        mode__ = cv2.IMREAD_COLOR
    elif (mode == "GRAY"):
        mode__ = cv2.IMREAD_GRAYSCALE

    img = cv2.imread("input_images/" + name, mode__)

    # if (img.mode is not 'RGB'):
    #     raise NotImplementedError()

    # im = np.array(img)

    elapsed_time = time.time() - start_time
    if (ELAPSED_TIME_OPT):
        print("Elapsed Time of imgload_cv : %d (sec)" %elapsed_time)

    return img

def add_salt_pepper_noise(X_imgs = None, amount = 0.01, mode = 'RGB'):
    start_time = time.time()
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
    elapsed_time = time.time() - start_time
    if (ELAPSED_TIME_OPT):
        print("Elapsed Time of add_salt_pepper_noise : %d (sec)" %elapsed_time)

    return X_imgs_copy

def salt_and_pepper(image, p):
    start_time = time.time()
    output = np.zeros(image.shape, dtype = np.uint8)
    thres = 1 - p

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):
            rdn = random()
            if (rdn < p):
                output[i][j] = 0
            elif (rdn > thres):
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]

    elapsed_time = time.time() - start_time
    if (ELAPSED_TIME_OPT):
        print("Elapsed Time of salt_and_pepper : %d (sec)" %elapsed_time)

    return output

def salt_and_pepper_fast(image, noise_typ, amount):
    start_time = time.time()
    np.random.seed(123)

    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        #var = 0.1
        #sigma = var**0.5
        gauss = np.random.normal(mean,1,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss

        elapsed_time = time.time() - start_time
        if (ELAPSED_TIME_OPT):
            print("Elapsed Time of salt_and_pepper_fast : %d (sec)" %elapsed_time)

        return noisy

    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = image

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0

        elapsed_time = time.time() - start_time
        if (ELAPSED_TIME_OPT):
            print("Elapsed Time of salt_and_pepper_cv : %d (sec)" %elapsed_time)

        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)

        elapsed_time = time.time() - start_time
        if (ELAPSED_TIME_OPT):
            print("Elapsed Time of salt_and_pepper_cv : %d (sec)" %elapsed_time)

        return noisy

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss

        elapsed_time = time.time() - start_time
        if (ELAPSED_TIME_OPT):
            print("Elapsed Time of salt_and_pepper_cv : %d (sec)" %elapsed_time)

        return noisy

def salt_and_pepper_gray(img, p):
    org_shape = img.shape
    wsize = img.shape[0] * img.shape[1]
    img = img.reshape(wsize)

    thres = 1 - p
    rnd_array = np.random.random(size = wsize)

    img[rnd_array < p] = 0
    img[rnd_array > thres] = 255

    return img.reshape(org_shape)
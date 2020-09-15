import numpy as np
import time

ELAPSED_TIME_OPT = True

def hw_RSEPD(input_image = None, Ts = 20):
    start_time = time.time()
    rowsize = input_image.shape[0]
    colsize = input_image.shape[1]
    denoised_image = np.zeros(input_image.shape,)

    # padIm = np.pad(input_image, ([1, 1], [1, 1]), 'symmetric')
    # padIm = np.pad(input_image, ([1, 1], [1, 1], [0, 0]), 'symmetric')
    padIm = np.pad(input_image, [1, 1], 'symmetric')
    padIm = padIm.astype(np.float64)

    row_buffer = np.zeros(colsize,)

    for i in range(1, rowsize + 1):
        for j in range(1, colsize + 1):
            ## Extreme Data Detector
            MINinW = 0.
            MAXinW = 255.
            pi = 0

            # f_bar = padIm[i, j]
            f_bar = 0
            flag1 = padIm[i, j] == MINinW
            flag2 = padIm[i, j] == MAXinW

            if ((padIm[i, j] == MINinW) or (padIm[i, j] == MAXinW)):
                pi = 1 # Noisy pixel

            if (pi == 0): # If not noisy pixel
                f_bar = padIm[i, j]

            else:
                b = 0

                if ((padIm[i + 1, j] == MINinW) or padIm[i + 1, j] == MAXinW):
                    b = 1 # Check surrounding pixel

                    ### Edge-Oriented Noise filter
                    if (b == 1): # If surrounding pixel is noisy
                        if ((padIm[i - 1, j - 1] == MINinW) and (padIm[i - 1, j] == MINinW) and (padIm[i - 1, j + 1] == MINinW)):
                            f_hat = MINinW
                        elif ((padIm[i - 1, j - 1] == MAXinW) and (padIm[i - 1, j] == MAXinW) and (padIm[i - 1, j + 1] == MAXinW)):
                            f_hat = MAXinW
                        else:
                            f_hat = ((padIm[i - 1, j - 1]) + 2 * (padIm[i - 1, j]) + (padIm[i - 1, j + 1])) / 4

                else: # If surrounding pixel is not noisy
                    Da = abs(padIm[i - 1, j - 1] - padIm[i + 1, j])
                    Db = abs(padIm[i - 1, j]     - padIm[i + 1, j])
                    Dc = abs(padIm[i - 1, j + 1] - padIm[i + 1, j])

                    f_hat_Da = (padIm[i - 1, j - 1] + padIm[i + 1, j]) / 2
                    f_hat_Db = (padIm[i - 1, j]     + padIm[i + 1, j]) / 2
                    f_hat_Dc = (padIm[i - 1, j + 1] + padIm[i + 1, j]) / 2

                    D = np.array([Da, Db, Dc])
                    Dmin = np.min(D)

                    if (Dmin == Da):
                        f_hat = f_hat_Da
                    elif (Dmin == Db):
                        f_hat = f_hat_Db
                    else:
                        f_hat = f_hat_Dc

                # Impulse Arbiter
                if (abs(padIm[i, j] - f_hat) > Ts):
                    f_bar = f_hat
                else:
                    f_bar = padIm[i, j]

            row_buffer[j - 1] = f_bar

        row_buffer = np.round(row_buffer, 0)
        denoised_image[i - 1, :] = row_buffer
        # temp = np.pad(row_buffer, [1], 'symmetric')
        padIm[i, :] = np.pad(row_buffer, [1], 'symmetric')

    print("hw_RSEPD end")
    denoised_image = np.clip(denoised_image, 0, 255)
    denoised_image = denoised_image.astype(np.uint8)
    elapsed_time = time.time() - start_time
    if (ELAPSED_TIME_OPT):
        print("Elapsed Time of hw_RSEPD : %d (sec)" %elapsed_time)
    return denoised_image

def hw_RSEPD_fast(input_image = None, Ts = 20):
    start_time = time.time()
    num_pad = 1
    # denoised_image = np.zeros(input_image.shape,)

    # padIm = np.pad(input_image, ([1, 1], [1, 1]), 'symmetric')
    # padIm = np.pad(input_image, ([1, 1], [1, 1], [0, 0]), 'symmetric')
    padIm = np.pad(input_image, [num_pad, num_pad], 'symmetric')
    padIm = padIm.astype(np.float64)
    rowsize = padIm.shape[0]
    colsize = padIm.shape[1]
    row_buffer = np.zeros((colsize,), dtype = np.float64)

    padIm = padIm.flatten()
    length = (rowsize  * colsize)
    MINinW = 0.
    MAXinW = 255.

    for i in range((colsize + 1), ((rowsize - 1) * colsize), 1):

        ## Extreme Data Detector
        (x, y) = ((i % colsize), (i // colsize))
        # print("Current index : [%d] // [%d] [%d]" %(i, x, y))

        # continue loop at padded pixels
        if ((y == 0) or (y == (rowsize - 1)) or (x == 0) or (x == (colsize - 1))):
            row_buffer[x] = padIm[i]

            if ((x - 1) == (colsize - 2)):
                padIm[((y * colsize) + 1): (((y + 1) * colsize) - 1)] = np.round(row_buffer, 0)[1: -1]
                row_buffer[:] = 0.0

            continue

        pi = 0
        f_bar = 0
        # flag1 = (padIm[i] == MINinW)
        # flag2 = (padIm[i] == MAXinW)

        if ((padIm[i] == MINinW) or (padIm[i] == MAXinW)):
            pi = 1

        if (pi == 0):
            f_bar = padIm[i]

        else:
            b = 0

            if ((padIm[i + colsize] == MINinW) or (padIm[i + colsize] == MAXinW)):
                b = 1

                if (b == 1):
                    if ((padIm[i - colsize -1] == MINinW) and (padIm[i - colsize] == MINinW) or (padIm[i - colsize + 1] == MINinW)):
                        f_hat = MINinW
                    elif ((padIm[i - colsize -1] == MAXinW) and (padIm[i - colsize] == MAXinW) or (padIm[i - colsize + 1] == MAXinW)):
                        f_hat = MAXinW
                    else:
                        f_hat = ((padIm[i - colsize - 1]) + 2 * (padIm[i - colsize]) + (padIm[i - colsize + 1])) / 4

            else:
                Da = abs(padIm[i - colsize - 1] - padIm[i + colsize])
                Db = abs(padIm[i - colsize] - padIm[i + colsize])
                Dc = abs(padIm[i - colsize + 1] - padIm[i + colsize])

                f_hat_Da = (padIm[i - colsize - 1] + padIm[i + colsize]) / 2
                f_hat_Db = (padIm[i - colsize] + padIm[i + colsize]) / 2
                f_hat_Dc = (padIm[i - colsize + 1] + padIm[i + colsize]) / 2

                D = np.array([Da, Db, Dc])
                Dmin = np.min(D)

                if (Dmin == Da):
                    f_hat = f_hat_Da
                elif (Dmin == Db):
                    f_hat = f_hat_Db
                else:
                    f_hat = f_hat_Dc

            # Impulse Arbiter
            if (abs(padIm[i] - f_hat) > Ts):
                f_bar = f_hat
            else:
                f_bar = padIm[i]

        row_buffer[x] = f_bar

        # if (x == (colsize - 2)):
        #     padIm[((y * colsize) + 1) : (((y + 1) * colsize) - 1)] = np.round(row_buffer, 0)[1 : -1]
        #     row_buffer[:] = 0.0

    elapsed_time = time.time() - start_time
    denoised_image = padIm.reshape([rowsize, colsize])
    denoised_image = denoised_image[1 : -1, 1: -1]
    if (ELAPSED_TIME_OPT):
        print("Elapsed Time of hw_RSEPD_fast : %d (sec)" %elapsed_time)

    return denoised_image

def paper_jrt(input_image = None, N = 4):
    start_time = time.time()
    # truncRed = 0
    # truncGreen = 0
    # truncBlue = 0
    input_image = input_image.astype(np.float64)
    output_image = np.zeros(input_image.shape)
    rowsize = (input_image.shape[0]) // N

    now = 0
    # gain = 0
    gain = np.zeros([input_image.shape[0], input_image.shape[2]], dtype = np.float64)

    for i in range(1, rowsize + 1):
        before = now
        # temp1 = input_image[:, N * (i - 1) : N * i, :]
        # temp = np.sum(input_image[:, N * (i - 1) : N * i, :], 1)
        now = np.squeeze(np.sum(input_image[:, N * (i - 1) : N * i, :], 1))

        if (i == 1):
            gain = now
        else:
            gain = gain + np.abs(before - now)
        # gain_view = np.sum(gain, 0)
        a = np.sum(gain, 0)

    finalGain = np.sum(gain, 0)

    som = np.sqrt((finalGain[0] ** 2) + (finalGain[1] ** 2) + (finalGain[2] ** 2))

    gain_R = 1 / (finalGain[0] / som)
    gain_G = 1 / (finalGain[1] / som)
    gain_B = 1 / (finalGain[2] / som)

    output_image[:, :, 0] = gain_R * input_image[:, :, 0]
    output_image[:, :, 1] = gain_G * input_image[:, :, 1]
    output_image[:, :, 2] = gain_B * input_image[:, :, 2]

    ## Value Check!!
    # is_valid = output_image[output_image < 0]
    output_image = np.clip(output_image, 0.0, 255.0)
    output_image = np.uint8(output_image)
    elapsed_time = time.time() - start_time
    if (ELAPSED_TIME_OPT):
        print("Elapsed Time of paper_jrt : %d (sec)" %elapsed_time)

    print("hw_JRT end")
    return output_image
import numpy as np

def hw_RSEPD(input_image = None, Ts = 0):
    rowsize = input_image.shape[0]
    colsize = input_image.shape[1]
    denoised_image = np.zeros(input_image.shape,)

    padIm = np.pad(input_image, [1, 1], 'symmetric')

    row_buffer = np.zeros(colsize,)

    for i in range(1, rowsize + 1):
        for j in range(2, colsize + 1):
            ## Extreme Data Detector
            MINinW = 0
            MAXinW = 255
            pi = 0

            if ((padIm[i, j] == MINinW) or (padIm[i, j] == MAXinW)):
                pi = 1 # Noisy pixel

            if (pi == 1): # If not noisy pixel
                f_bar = padIm[i, j]

            else:
                b = 0

                if ((padIm[i + 1, j] == MINinW) or padIm[i + 1, j] == MAXinW):
                    b = 1 # Check surrounding pixel

                    ### Edge-Oriented Noise filter
                    if (b == 1): # If surrounding pixel is noisy
                        if ((padIm[i - 1, j - 1] == MINinW) and (padIm[i - 1, j] == MINinW) and (padIm[i - 1, j + 1] == MINinW)):
                            f_hat = MINinW
                        elif ((padIm[i - 1, j - 1] == MAXinW) and (padIm[i - 1, j] == MAXinW) and (padIm[i - 1, j + 1] == MINinW)):
                            f_hat = MAXinW
                        else:
                            f_hat = ((padIm[i - 1, j - 1]) + 2 * (padIm[i - 1, j]) + (padIm[i - 1, j + 1])) / 4

                    else: # If surrounding pixel is not noisy
                        Da = abs(padIm[i - 1, j - 1] - padIm[i + 1, j])
                        Db = abs(padIm[i - 1, j] -     padIm[i + 1, j])
                        Dc = abs(padIm[i - 1, j + 1] - padIm[i + 1, j + 1])

                        f_hat_Da = (padIm[i - 1, j - 1] + padIm[i + 1, j]) / 2
                        f_hat_Db = (padIm[i - 1, j] +     padIm[i + 1, j]) / 2
                        f_hat_Dc = (padIm[i - 1, j + 1] + padIm[i + 1, j]) / 2

                        D = [Da, Db, Dc]
                        Dmin = min(D)

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
        padIm[i, :] = np.pad(row_buffer, [1], 'symmetric')

    print("hw_RSEPD end")

    return denoised_image


def paper_jrt(input_image = None, N = 4):
    # truncRed = 0
    # truncGreen = 0
    # truncBlue = 0

    output_image = np.zeros(input_image.shape)
    rowsize = (input_image.shape[0]) // N

    now = 0

    for i in range(rowsize):
        before = now
        now = np.squeeze(sum(input_image[:, N * (i - 1) : N * i, :], 1))

        if (i == 1):
            gain = now
        else:
            gain = gain + abs(before - now)

    finalGain = sum(gain, 0)

    som = np.sqrt((finalGain[0] ** 2) + (finalGain[1] ** 2) + (finalGain[2] ** 2))

    gain_R = 1 / (finalGain[0] / som)
    gain_G = 1 / (finalGain[1] / som)
    gain_B = 1 / (finalGain[2] / som)

    output_image[:, :, 0] = gain_R
    output_image[:, :, 1] = gain_G
    output_image[:, :, 2] = gain_B

    return output_image
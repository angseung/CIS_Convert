import cv2
import numpy as np
import time

ELAPSED_TIME_OPT = False

"""
CIS Functions
"""

def salt_and_pepper(img, p):
    org_shape = img.shape
    wsize = img.shape[0] * img.shape[1]
    img = img.reshape(wsize)

    thres = 1 - p
    rnd_array = np.random.random(size=wsize)

    img[rnd_array < p] = 0
    img[rnd_array > thres] = 255

    return img.reshape(org_shape)

def hw_RSEPD_fast_HT(input_image = None, Ts = 20):
    # start_time = time.time()
    # rowsize = input_image.shape[0]
    # colsize = input_image.shape[1]
    rowsize, colsize = input_image.shape
    # denoised_image = np.zeros((rowsize, colsize), dtype = np.float64)   # TODO: ZEROS_LIKE

    padIm = np.pad(input_image, [1, 1], 'symmetric')
    padIm = padIm.astype(np.float64)

    # row_buffer = np.zeros(colsize, )
    MINinW = 0.
    MAXinW = 255.

    for i in range(1, rowsize + 1):
        noisyLine = (padIm[i, 1:-1] != MAXinW) & (padIm[i, 1:-1] != MINinW)

        f_hat_line = np.zeros(colsize,)
        f_bar_line = padIm[i, 1:-1]
        orig_line = padIm[i, 1:-1]

        temp = padIm[i + 1, 1 : -1]
        DaLine = (abs(padIm[i - 1, 0 : -2] - temp))
        DbLine = (abs(padIm[i - 1, 1 : -1] - temp))
        DcLine = (abs(padIm[i - 1, 2 :   ] - temp))

        temp = padIm[i + 1, 1 : -1]
        f_hat_DaLine = (padIm[i - 1, 0 : -2] + temp)/2
        f_hat_DbLine = (padIm[i - 1, 1: -1]  + temp)/2
        f_hat_DcLine = (padIm[i - 1, 2:]     + temp)/2

        DLine = np.vstack((DaLine, DbLine, DcLine)) # comparing stack function

        Dmin = np.argmin(DLine, 0)  # min
        zeroIdx = Dmin == 0
        oneIdx = Dmin == 1
        twoIdx = Dmin == 2
        f_hat_line[zeroIdx] = f_hat_DaLine[zeroIdx]
        f_hat_line[oneIdx] = f_hat_DbLine[oneIdx]
        f_hat_line[twoIdx] = f_hat_DcLine[twoIdx]

        noisyRefLine =  (padIm[i+1, 1:-1] == MAXinW) | (padIm[i+1, 1:-1] == MINinW)
        meanLine = (padIm[i-1,0:-2] + 2*padIm[i-1,1:-1] + padIm[i-1,2:])/4
        f_hat_line[noisyRefLine] = meanLine[noisyRefLine]

        thr = abs(padIm[i, 1:-1] - f_hat_line)
        comp_thr = thr > Ts
        f_bar_line[comp_thr] = f_hat_line[comp_thr]
        f_bar_line[noisyLine] = orig_line[noisyLine]

        row_buffer = np.clip(f_bar_line, 0, 255)
        # denoised_image[i - 1, :] = row_buffer
        padIm[i, :] = np.pad(row_buffer, [1], 'symmetric')

    # print("hw_RSEPD_fast_HT end")
    # denoised_image = np.clip(denoised_image, 0, 255)
    # denoised_image = denoised_image.astype(np.uint8)
    # elapsed_time = time.time() - start_time

    # if (ELAPSED_TIME_OPT):
    #     print("Elapsed Time of hw_RSEPD_fast_HT : %d (sec)" %elapsed_time)
    denoised_image = padIm[1 : -1, 1 : -1].astype(np.uint8)

    return denoised_image

def white_balance(input_image=None, N=4):
    # truncRed = 0
    # truncGreen = 0
    # truncBlue = 0
    input_image = input_image.astype(np.float64)
    output_image = np.zeros(input_image.shape)
    rowsize = (input_image.shape[0]) // N

    now = 0
    gain = np.zeros([input_image.shape[0], input_image.shape[2]], dtype=np.float64)

    for i in range(1, rowsize + 1):
        before = now
        now = np.squeeze(np.sum(input_image[:, N * (i - 1): N * i, :], 1))

        if (i == 1):
            gain = now
        else:
            gain = gain + np.abs(before - now)

    finalGain = np.sum(gain, 0)

    som = np.sqrt((finalGain ** 2).sum())

    gain_R = 1 / (finalGain[0] / som)
    gain_G = 1 / (finalGain[1] / som)
    gain_B = 1 / (finalGain[2] / som)

    output_image[:, :, 0] = gain_R * input_image[:, :, 0]
    output_image[:, :, 1] = gain_G * input_image[:, :, 1]
    output_image[:, :, 2] = gain_B * input_image[:, :, 2]

    output_image = np.clip(output_image, 0, 255.0)
    output_image = np.uint8(output_image)

    return output_image

"""
Entry point Main
"""

cam_src = "nvarguscamerasrc num_buffer=1 ! " \
          "video/x-raw(memory:NVMM), width=(int)400, height=(int)300, format=(string)NV12, framerate=(fraction)30/1 ! " \
          "nvvidconv ! video/x-raw,format=(string)BGRx ! " \
          "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
font = cv2.FONT_HERSHEY_PLAIN

capture = cv2.VideoCapture(cam_src)
# capture = cv2.VideoCapture('/dev/video0',cv2.CAP_FFMPEG)


while True:
    start = time.time()
    ret, frame = capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noisy_frame = salt_and_pepper(gray_frame, 0.004)
    # wbframe = white_balance(frame)     # TODO : ***
    nrframe = hw_RSEPD_fast_HT(noisy_frame, Ts=20) # TODO : ***

    nframeRs = cv2.cvtColor(noisy_frame, cv2.COLOR_GRAY2BGR)
    nrframeRs = cv2.cvtColor(nrframe, cv2.COLOR_GRAY2BGR)

    # vidBuf_upper = np.concatenate((frame, nframeRs), axis=1)
    # vidBuf_lower = np.concatenate((wbframe, nrframeRs), axis=1)

    vidBuf = np.vstack((frame, nframeRs, nrframeRs))
    # vidBuf = np.concatenate((frame, nframeRs, nrframeRs), axis=0)

    fpstxt = "Estimated frames per second : {0}".format(1 / (time.time() - start))
    # fps2txt = "Estimated time : {0}".format((time.time() - start))
    cv2.putText(vidBuf, fpstxt, (11, 20), font, 1.0, (50, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(vidBuf, fps2txt, (11, 120), font, 1.0, (50, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("VideoFrame", vidBuf)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()

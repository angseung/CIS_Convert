import sys
import argparse
import cv2
import numpy as np
import time

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

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", dest="video_device",
                        help="Video device # of USB webcam (/dev/video?) [0]",
                        default=0, type=int)
    arguments = parser.parse_args()
    return arguments


# Use a Jetson TX1 or TX2 camera in the Jetson AGX Xavier Camera Slot
def open_onboard_camera():
    return cv2.VideoCapture("nvarguscamerasrc sensor_mode=1 ! " \
          "video/x-raw(memory:NVMM), width=(int)400, height=(int)300, format=(string)NV12, framerate=(fraction)30/1 ! " \
          "nvvidconv ! video/x-raw,format=(string)BGRx ! " \
          "videoconvert ! video/x-raw, format=(string)BGR ! appsink")



# Open an external usb camera /dev/videoX
def open_camera_device(device_number):
    return cv2.VideoCapture(device_number)


def read_cam(video_capture):
    if video_capture.isOpened():
        windowName = "CannyDemo"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 1280, 720)
        cv2.moveWindow(windowName, 0, 0)
        cv2.setWindowTitle(windowName, "Camera Show")
        showWindow = 3  # Show all stages
        showHelp = True
        font = cv2.FONT_HERSHEY_PLAIN
        # helpText = "'Esc' to Quit, '1' for Camera Feed, '2' for Canny Detection, '3' for All Stages. '4' to hide help"
        edgeThreshold = 40
        showFullScreen = False

        while True:
            if cv2.getWindowProperty(windowName, 0) < 0:  # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;

            (ret_val, frame) = video_capture.read();

            if frame is None:
                continue;

            if showWindow == 3:  # Show Camera Display ONLY
                # Composite the 2x2 window
                # Feed from the camera is RGB, the others gray
                # To composite, convert gray images to color.
                # All images must be of the same type to display in a window
                # frameRs = cv2.resize(frame, (640, 360))
                # hsvRs = cv2.resize(frame1, (640, 360))
                vidBuf = frame
                # blurRs = cv2.resize(blur, (640, 360))

            if showWindow == 1:  # Show NR Test
                start = time.time()

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                noisy_frame = salt_and_pepper(gray_frame, 0.004)
                nrframe = hw_RSEPD_fast_HT(noisy_frame, Ts = 20)

                nframeRs = cv2.cvtColor(noisy_frame, cv2.COLOR_GRAY2BGR)
                nrframeRs = cv2.cvtColor(nrframe, cv2.COLOR_GRAY2BGR)

                vidBuf = np.hstack((frame, nframeRs, nrframeRs))
                displayBuf = vidBuf
                end_time = time.time()

            elif showWindow == 2:  # Show AWB Test
                start = time.time()

                wbframe = white_balance(frame)
                vidBuf = np.hstack((frame, wbframe))
                displayBuf = vidBuf
                end_time = time.time()

            elif showWindow == 3:  # Show All Stages
                displayBuf = frame

            if ((showHelp == True) and ((showWindow == 1) or (showWindow == 2))):
                fpstxt = "Estimated frames per second : {0}".format(1 / (end_time - start))
                cv2.putText(vidBuf, fpstxt, (11, 20), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(windowName, displayBuf)

            key = cv2.waitKey(10)

            if key == 27:  # Check for ESC key
                cv2.destroyAllWindows()
                break;

            elif key == 49:  # 1 key, show frame
                cv2.setWindowTitle(windowName, "NR TEST")
                showWindow = 1

            elif key == 50:  # 2 key, show Canny
                cv2.setWindowTitle(windowName, "AWB TEST")
                showWindow = 2

            elif key == 51:  # 3 key, show Stages
                cv2.setWindowTitle(windowName, "Camera Show")
                showWindow = 3

    else:
        print("camera open failed")


if __name__ == '__main__':
    arguments = parse_cli_args()
    print("Called with args:")
    print(arguments)
    print("OpenCV version: {}".format(cv2.__version__))
    print("Device Number:", arguments.video_device)
    print(cv2.getBuildInformation())
    if arguments.video_device == 0:
        print("Using on-board camera")
        video_capture = open_onboard_camera()
    else:
        video_capture = open_camera_device(arguments.video_device)
        # Only do this on external cameras; onboard will cause camera not to read
        video_capture.set(cv2.CAP_PROP_FPS, 30)

    read_cam(video_capture)
    video_capture.release()
    cv2.destroyAllWindows()
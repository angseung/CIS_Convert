import caffe
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QSize, Qt, QThread, QEventLoop, QTimer
from PyQt5.QtGui import QTextCursor, QImage, QColor, QPixmap
from datetime import datetime
import numpy as np
from ai_core import trainingCore, testCore
from tools import topk, calSquare, caffeTools
from yolo_function import *

import cv2
import sys, os

import pyqtgraph as pg

_platform_ = sys.platform


class logMessage(QObject):
    """This is a logMessage object
        Author : seokhun.jeon@keti.re.kr
        Last Update : 04.Mar.2020
    """

    # Predefined Log Style
    log_style = {
        'Alert': "<font color=\"DeepPink\">",
        'Info': "<font color=\"Lime\">",
        'Warning': "<font color =\"Yellow\">",
        'Notify': "<font color=\"Aqua\">",
        'Normal': "<font color =\"White\">"
    }

    def __init__(self, widgets):
        """ UI constructor
        """
        super(logMessage, self).__init__()

        self.widgets = widgets

    def setLog(self, rd, type='Normal'):
        """Log message function
        :param rd: message
        :param type: message type [default : Normal]
        :return: None
        """
        self.widgets.log.moveCursor(QTextCursor.End)

        dateTimeObj = datetime.now()

        try:
            header = self.log_style[type]
        except Exception as e:
            print(e)
            header = self.log_style['Normal']

        self.widgets.log.textCursor().insertHtml(
            header + dateTimeObj.strftime("%H:%M:%S.%f")[:-3] + '\t' + str(rd) + '<br>')


class testMain(QObject):
    """This is caffe test main object
        Author : seokhun.jeon@keti.re.kr
        Last Update : 04.Mar.2020
    """
    log = pyqtSignal(str, str)
    status = pyqtSignal(str)

    def __init__(self, widgets):
        super().__init__()

        # Main parameter initialization
        self.deploy_path = None
        self.caffemodel_path = None
        self.net = None
        self.cam = None

        self.layer_item = {'Params': [], 'Blobs': []}

        self.widgets = widgets

        self.widgets.netinputconfig.load_button.clicked.connect(self.load_netinput)
        self.widgets.netinputconfig.cam_rescale.valueChanged.connect(self.set_cam_rescale)

        self.widgets.networkmonitor.layer_name.currentTextChanged.connect(self.__layer_channel_change)
        self.widgets.networkmonitor.layer_type.currentTextChanged.connect(self.__layer_name_change)
        self.widgets.networkmonitor.layer_channel.valueChanged.connect(self.__update_hidden_result)

        self.widgets.test_configs.run_test.clicked.connect(self.run_test)

    def run_test(self):
        iw = self.widgets.netinputconfig
        tc = self.widgets.test_configs
        run_flag = False

        if iw.input_mode.currentText() == 'Image':
            # if self.cam is not None:
            #     self.cam.toggle_status()

            run_flag = True
            if iw.input_type[0].isChecked():
                if self.cam._pause:
                    tc.run_test.setText('Release Camera')
                else:
                    tc.run_test.setText('Run Test')
                    run_flag = False
            else:
                tc.run_test.setText('Run Test')

        elif iw.input_mode.currentText() == 'Video':
            tc.run_test.setText('Run Test')
            self.cam.net_test()

            if self.cam.net_start:
                tc.run_test.setText('Test OFF')
            else:
                tc.run_test.setText('Test ON')

        if run_flag:
            if self.net is not None:

                if not tc.test_mode.isChecked():
                    # object detection mode (YOLO network's input range 0~1 float)
                    self.last_frame = self.last_frame.astype(float)
                    self.last_frame /= 255

                self.log.emit(' ::: Caffe Model Forward Start :::', 'Info')
                self.tc = testCore(self.last_frame, self.net)
                self.tc.start()
                self.tc.net_out.connect(self.__net_update)

            else:
                self.log.emit('Caffe Model is NOT loaded')

    def load_netinput(self):

        self.__set_test_network()

        iw = self.widgets.netinputconfig

        # TODO : CHECK THIS ROUTINE!!
        if self.cam is not None:
            self.cam.test_flag = False
            self.camThread.terminate()

        if iw.input_mode.currentText() == 'Image':
            self.log.emit('+++ Image Mode +++', 'Info')
            if iw.input_type[0].isChecked():
                self.log.emit('- From File', 'Info')
                inputfile_path = iw.inputfile_path.text()
                self.log.emit('- Path : %s' % inputfile_path, 'Info')

                if inputfile_path == '':
                    self.log.emit('File is NOT selected!!', 'Alert')
                    # raise AttributeError('File is NOT selected!!')
                else:
                    self.frame = caffe.io.load_image(inputfile_path) * 255
                    self.frame = self.frame.astype(np.uint8)

                    self.__input_update(self.frame)

            elif iw.input_type[1].isChecked():
                self.log.emit('- From Video', 'Info')
                self.log.emit('Initialize Camera Interface...', 'Info')

                cam_num = self.widgets.netinputconfig.intput_cam.currentIndex()
                res_info = self.widgets.netinputconfig.cam_resolution.currentText()
                rescale_value = self.widgets.netinputconfig.cam_rescale.value()

                if res_info == '1080p':
                    vResoultion = (1920, 1080)
                elif res_info == '720p':
                    vResoultion = (1280, 720)
                elif res_info == '480p':
                    vResoultion = (640, 480)
                self.log.emit('- CAM Resolution %s' % str(vResoultion), 'Info')
                self.log.emit('- CAM Rescale %d' % rescale_value, 'Info')

                if _platform_.__contains__('linux'):
                    cam_num = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink" % (
                        vResoultion[0], vResoultion[1])

                self.log.emit('CAM%s' % cam_num, 'Info')

                self.cam = Camera(cam_num, vResoultion, rescale_value)
                self.cam.log.connect(lambda v0, v1: self.log.emit(v0, v1))
                self.camThread = QThread()
                self.camThread.start()
                self.cam.moveToThread(self.camThread)
                self.cam.video_signal.connect(self.__input_update)
                self.cam.startVideo()

        elif iw.input_mode.currentText() == 'Video':
            self.log.emit('+++ Video Mode +++', 'Info')

            if self.net is not None:
                res_info = self.widgets.netinputconfig.cam_resolution.currentText()
                rescale_value = self.widgets.netinputconfig.cam_rescale.value()

                if res_info == '1080p':
                    vResoultion = (1920, 1080)
                elif res_info == '720p':
                    vResoultion = (1280, 720)
                elif res_info == '480p':
                    vResoultion = (640, 480)

                if iw.input_type[0].isChecked():
                    self.log.emit('- From File', 'Info')
                    inputfile_path = iw.inputfile_path.text()
                    self.log.emit('- Path : %s' % inputfile_path, 'Info')

                    self.log.emit('- CAM Resolution %s' % str(vResoultion), 'Info')
                    self.log.emit('- CAM Rescale %d' % rescale_value, 'Info')

                    self.cam = Camera(inputfile_path, vResoultion, rescale_value, self.net)

                elif iw.input_type[1].isChecked():
                    self.log.emit('- From Video', 'Info')
                    self.log.emit('Initialize Camera Interface...', 'Info')

                    self.log.emit('- CAM Resolution %s' % str(vResoultion), 'Info')
                    self.log.emit('- CAM Rescale %d' % rescale_value, 'Info')

                    cam_num = self.widgets.netinputconfig.intput_cam.currentIndex()

                    if _platform_.__contains__('linux'):
                        cam_num = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink" % (
                            vResoultion[0], vResoultion[1])

                    self.log.emit('CAM%s' % cam_num, 'Info')

                    self.cam = Camera(cam_num, vResoultion, rescale_value, self.net)

                self.camThread = QThread()
                self.camThread.start()
                self.cam.moveToThread(self.camThread)
                self.cam.video_signal.connect(self.__input_update)
                self.cam.startVideo()
                self.cam.net_out.connect(self.__net_update)
                self.cam.log.connect(lambda v0, v1: self.log.emit(v0, v1))

            else:
                self.log.emit('Network is not Loaded!!', 'Alert')

    def __net_update(self, net):
        self.net = net

        tc = self.widgets.test_configs

        if tc.test_mode.isChecked():
            self.log.emit(' ::: Classification Mode :::', 'Info')
            self.proba = self.net.blobs['prob'].data[0]

            self.__update_top_k_error(self.param['Out_Topnum'].value())
        else:
            self.log.emit(' ::: Object Detection Mode :::', 'Info')

            out1d = reorderOutput(self.net.blobs['conv9'].data.flatten())

            img_height, img_width, _ = self.last_frame.shape
            results = interpretOutputV2(out1d, img_width, img_height)

            for result_log in results:
                self.log.emit(str(result_log), 'Info')

            img_cp = self.last_frame.copy()
            img_cp = (img_cp * 255).astype(np.uint8)

            for i in range(len(results)):
                conf = results[i].conf
                xmin = results[i].xmin
                xmax = results[i].xmax
                ymin = results[i].ymin
                ymax = results[i].ymax

                if xmin < 0.0:
                    xmin = 0.0
                if ymin < 0.0:
                    ymin = 0.0
                if xmax > img_width:
                    xmax = img_width
                if ymax > img_height:
                    ymax = img_height

                scolor = cmappp[classes_name.index(results[i].object_class)]
                img_cp = cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), scolor, 2)
                text = results[i].object_class + ' : %.2f' % results[i].conf
                img_cp = cv2.putText(img_cp, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, scolor, 2)

                self.__input_update(img_cp)

                self.log.emit(' ::: Object Detection Done :::', 'Info')

    def __input_update(self, img):
        height, width, channel = img.shape
        bpl = channel * width

        frame = QImage(img.data, width, height, bpl, QImage.Format_RGB888)

        self.last_frame = img
        self.widgets.networkmonitor.input_image_display.setPixmap(QPixmap.fromImage(frame))

    def set_cam_rescale(self):
        rescale_value = self.widgets.netinputconfig.cam_rescale.value()
        if self.cam is not None:
            self.cam.setRescale(rescale_value)
            self.log.emit('Change Cam reScale Factor %d' % rescale_value, 'Info')
        else:
            self.log.emit('Cam is not connected!!', 'Alert')

    def __set_test_network(self):
        self.deploy_path = self.widgets.netinputconfig.deploy_path.text()
        self.caffemodel_path = self.widgets.netinputconfig.caffemodel_path.text()

        if self.deploy_path is not None and self.caffemodel_path is not None:

            if self.widgets.test_configs.mode.currentText() == 'CPU':
                self.log.emit('Test Mode : CPU!!', 'Info')
                caffe.set_mode_cpu()

            elif self.widgets.test_configs.mode.currentText() == 'GPU':
                self.log.emit('Test Mode : GPU!!', 'Info')
                caffe.set_mode_gpu()
                caffe.set_device(0)

            self.net = caffe.Net(self.deploy_path, self.caffemodel_path, caffe.TEST)
            self.__reset_hidden_monitoring_widget()

            self.log.emit('Network Load Done', 'Info')
        else:

            self.log.emit('Network Load Failed', 'Info')

    def __reset_hidden_monitoring_widget(self):
        self.layer_item['Params'] = list(self.net.params.keys())
        self.layer_item['Blobs'] = list(self.net.blobs.keys())

        self.__layer_name_change(self.widgets.networkmonitor.layer_type.currentText())

    def __layer_name_change(self, curtext):

        nm = self.widgets.networkmonitor

        try:
            nm.layer_name.clear()
        except Exception as e:
            print(e)

        nm.layer_name.addItems(self.layer_item[curtext])
        nm.layer_name.setCurrentIndex(0)

        self.__update_hidden_result(0)

    def __layer_channel_change(self, curtext):

        nm = self.widgets.networkmonitor

        if nm.layer_type.currentText() == 'Params':
            max_channel = self.net.params[nm.layer_name.currentText()][0].shape[0]

        elif nm.layer_type.currentText() == 'Blobs':
            max_channel = self.net.blobs[nm.layer_name.currentText()].shape[0]

        nm.layer_channel.setMaximum(max_channel - 1)
        nm.layer_channel.setMinimum(0)
        nm.layer_channel.setValue(0)

        self.__update_hidden_result(0)

    def __update_hidden_result(self, idx):

        nm = self.widgets.networkmonitor

        if nm.layer_type.currentText() == 'Params':
            hidden_data = self.net.params[nm.layer_name.currentText()][0].data[idx].copy()

        elif nm.layer_type.currentText() == 'Blobs':
            hidden_data = self.net.blobs[nm.layer_name.currentText()].data[idx].copy()

        nm.hidden_dsiplay.setTableConfiguration(*calSquare(hidden_data.shape[0], 1))
        nm.hidden_dsiplay.setData(hidden_data)

    def __update_top_k_error(self, k):

        label_path = self.widgets.test_configs.disp_label.text()
        if label_path != '':
            lb = open(label_path).read().split('\n')
            print(lb)
        else:
            lb = np.arange(len(self.net.blobs[self.layer_item['Blobs'][-1]]))
            lb = lb.astype(np.str)

        result = topk(self.proba, lb, k)

        self.log.emit('[Classification Result]')

        for i in range(k):
            self.log.emit('%.5f %s' % (result[i][0], result[i][1]))


class Camera(QObject):
    video_signal = pyqtSignal(np.ndarray)
    test_flag = True
    net_out = pyqtSignal(object)
    net_start = 0
    log = pyqtSignal(str, str)

    def __init__(self, cam_num, resolution, rescale, net=None):
        super().__init__()
        self.cam_num = cam_num
        self.cap = None
        self.net = net
        self.resolution = resolution
        self.rescale = rescale

        self.last_frame = np.zeros((1, 1))

        self._pause = False

    def setRescale(self, rescale):
        self.rescale = rescale

    def initialize(self):
        self.cap = cv2.VideoCapture(self.cam_num)

        if not _platform_.__contains__('linux'):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

    def rescale_frame(self, frame, percent=75):
        
        if _platform_.__contains__('linux'):
            if self.cam_num.__contains__('nvarguscamerasrc'):
                frame = cv2.flip(frame,0)
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def startVideo(self, opt='C'):
        self.log.emit('Camera initialization!!', 'Info')
        self.initialize()
        self.log.emit('Done', 'Info')
        while self.test_flag:
            ret, frame = self.cap.read()

            frame = self.rescale_frame(frame, self.rescale)

            # cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            color_swapped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.net_start:
                if self.net is not None:
                    caffe.set_mode_gpu()
                    caffe.set_device(0)
                    transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
                    transformer.set_transpose('data', (2, 0, 1))
                    start = datetime.now()
                    self.net.forward_all(data=np.asarray([transformer.preprocess('data', color_swapped_image / 255)]))
                    elapsed_time = datetime.now() - start

                    #cv2.putText(frame, 'elapsed: {:.2f}'.format(elapsed_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    #            (0, 255, 0),
                    #            2)

                    out1d = reorderOutput(self.net.blobs['conv9'].data.flatten())
                    # np.savetxt('conv9_output.txt',out1d, fmt='%f')

                    img_height, img_width, _ = color_swapped_image.shape

                    results = interpretOutputV2(out1d, img_width, img_height)
                    for result_log in results:
                        self.log.emit(str(result_log), 'Info')

                    img_cp = color_swapped_image.copy()

                    for i in range(len(results)):
                        conf = results[i].conf
                        xmin = int(results[i].xmin*0.95)
                        xmax = int(results[i].xmax*0.95)
                        ymin = int(results[i].ymin*0.90)
                        ymax = int(results[i].ymax*0.90)

                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if xmax > img_width:
                            xmax = img_width
                        if ymax > img_height:
                            ymax = img_height

                        scolor = cmappp[classes_name.index(results[i].object_class)]
                        try:
                            img_cp = cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), scolor, 2)

                            text = results[i].object_class + ' : %.2f' % results[i].conf
                            img_cp = cv2.putText(img_cp, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, scolor, 2)
                        except TypeError:
                            pass
                    self.video_signal.emit(img_cp)
                    self.net_out.emit(self.net)

            else:
                if not self._pause:
                    self.video_signal.emit(color_swapped_image)

                if not self.cap.isOpened():
                    print('Video End')
                    break

            loop = QEventLoop()
            QTimer.singleShot(10, loop.quit)
            loop.exec_()

        self.cap.release()

    def toggle_status(self):
        self._pause = not self._pause
        if self._pause:
            self.cond.wakeAll()

    def net_test(self):
        self.net_start = 1 - self.net_start
        print(self.net_start)

    def __str__(self):
        return 'OpenCV Camera {}'.format(self.cam_num)


class trainingMain(QObject):
    """This is caffe training main object
        Author : seokhun.jeon@keti.re.kr
        Last Update : 04.Mar.2020
    """

    log = pyqtSignal(str, str)
    status = pyqtSignal(str)

    def __init__(self, widgets):
        super().__init__()

        # Main parameter initialization
        self.solver = None
        self.training_thread = None
        self.widgets = widgets

        self.monwidget = {'a': self.widgets.networkmonitor.input_batch_number,
                          'b': self.widgets.networkmonitor.hidden_layername,
                          'c': self.widgets.networkmonitor.result_topknum}

        # tools
        self.caffe_tools = caffeTools()
        self.caffe_tools.log.connect(lambda c, d: self.log.emit(c, d))

        self.test_profiler = trainProfiler(self.widgets.networkmonitor.hidden_layerdisplay)

        self.profiler_thread = QThread()
        self.test_profiler.moveToThread(self.profiler_thread)
        self.profiler_thread.start()

        self.test_profiler.log.connect(lambda c, d: self.log.emit(c, d))
        self.test_profiler.time_tbl_update.connect(
            lambda a, b, c: self.widgets.solvermonitor.net_properties.setItem(a, b, QTableWidgetItem(c)))
        self.test_profiler.time_graph_update.connect(self.timegraphUpdate)
        self.test_profiler.perf_graph_update.connect(self.netgraphUpdate)

        self.test_profiler.input_image_update.connect(self.widgets.networkmonitor.input_image_display.setPixmap)

        self.test_profiler.input_blue_hist_update.connect(
            lambda v: self.widgets.networkmonitor.input_bluehist.setData(**v))
        self.test_profiler.input_red_hist_update.connect(
            lambda v: self.widgets.networkmonitor.input_redhist.setData(**v))
        self.test_profiler.input_green_hist_update.connect(
            lambda v: self.widgets.networkmonitor.input_greenhist.setData(**v))

        self.test_profiler.hidden_hist_update.connect(self.hidden_hist_update)
        self.test_profiler.output_barplot.connect(lambda v: self.widgets.networkmonitor.result_display.setData(**v))

        self.test_profiler.mon_update.connect(lambda v=self.monwidget: self.test_profiler.netMonitoring(**v))

        # Signal Slots
        self.widgets.control.run_button.clicked.connect(self.trainingOperation)
        self.widgets.control.savesolver_button.clicked.connect(self.save_solver)

    def trainingOperation(self):
        if self.solver is not None:
            if self.training_thread is None:
                mode = self.widgets.control.mode.currentText()
                training_iteration = int(self.widgets.control.iteration.text())
                self.training_thread = trainingCore(self.solver, training_iteration, mode=mode)

                self.training_thread.start()

                self.training_thread.training_progress.connect(self.widgets.control.progressbar.setValue)
                self.training_thread.log.connect(lambda c, d: self.log.emit(c, d))
                # self.training_thread.time_out.connect(self.timegraphUpdate)
                # self.training_thread.net_out.connect(self.netgraphUpdate)
                self.training_thread.net_out.connect(self.test_profiler.netUpdate)
                self.training_thread.val_out.connect(self.valgraphUpdate)
                self.training_thread.finished.connect(self.finish_solver)

                # self.training_thread.time_out.connect(self.test_profiler.timeUpdate)

                self.widgets.control.iteration.setEnabled(False)
                self.widgets.control.run_button.setText('Stop')
            else:
                if not self.training_thread.finish_flag:
                    self.training_thread.setFinish()
        else:
            self.log.emit('Caffe Model is NOT loaded', 'Alert')

    def finish_solver(self):
        self.log.emit('Training Done', 'Notify')
        self.training_thread.setFinish()

        self.widgets.control.run_button.setText('Run')
        self.widgets.control.iteration.setEnabled(True)

        del self.training_thread

        self.training_thread = None

    def hidden_hist_update(self, curve):
        for i in self.widgets.networkmonitor.hidden_histcurve:
            self.widgets.networkmonitor.hidden_histplot.removeItem(i)

        if len(self.widgets.networkmonitor.hidden_histcurve) >= 3:
            self.widgets.networkmonitor.hidden_histcurve.pop(0)
            self.widgets.networkmonitor.hidden_histcurve.append(curve)
        else:
            self.widgets.networkmonitor.hidden_histcurve.append(curve)

        for i in self.widgets.networkmonitor.hidden_histcurve:
            self.widgets.networkmonitor.hidden_histplot.addItem(i)

    @pyqtSlot(tuple)
    def valgraphUpdate(self, valdat):
        """network performance [validation result] update
        :param valdat:
        :return:
        """

        # Network Train Performance data
        self.test_perf_data_object['Iteration'].append(valdat[0])
        self.test_perf_data_object['test_acc'].append(valdat[1])

        self.widgets.solvermonitor.test_acc_plot.setData(x=self.test_perf_data_object['Iteration'],
                                                         y=self.test_perf_data_object['test_acc'],
                                                         clear=True)

    @pyqtSlot(tuple)
    def netgraphUpdate(self, perf_packet):
        """network parameter monitoring event loop
        :param solver: caffe solver object contained output and weights of each layer
        :return: graph output
        """

        # Network Train Performance data
        self.train_perf_data_object['Iteration'].append(perf_packet[0])
        self.train_perf_data_object['train_loss'].append(perf_packet[1])
        self.train_perf_data_object['train_acc'].append(perf_packet[2])

        self.widgets.solvermonitor.train_acc_plot.setData(x=self.train_perf_data_object['Iteration'],
                                                          y=self.train_perf_data_object['train_acc'],
                                                          clear=True)
        self.widgets.solvermonitor.train_loss_plot.setData(x=self.train_perf_data_object['Iteration'],
                                                           y=self.train_perf_data_object['train_loss'],
                                                           clear=True)

        # self.setMonitoringdata()

        # self.inoutUpdate()
        # self.hiddenUpdate()

    @pyqtSlot(tuple)
    def timegraphUpdate(self, time_packet):

        self.time_data_object['Iteration'].append(time_packet[0])
        self.time_data_object['Forward'].append(time_packet[1])
        self.time_data_object['Backward'].append(time_packet[2])
        self.time_data_object['Total'].append(time_packet[1] + time_packet[2])

        # Proc TIme Update
        self.widgets.solvermonitor.forward_time.setData(x=self.time_data_object['Iteration'],
                                                        y=self.time_data_object['Forward'],
                                                        clear=True)
        self.widgets.solvermonitor.backward_time.setData(x=self.time_data_object['Iteration'],
                                                         y=self.time_data_object['Backward'],
                                                         clear=True)
        self.widgets.solvermonitor.total_time.setData(x=self.time_data_object['Iteration'],
                                                      y=self.time_data_object['Total'],
                                                      clear=True)

        # Throughput update
        self.widgets.control.throughput.setText('%.3f' % (time_packet[1] + time_packet[2]))
        self.widgets.control.throughput.setCursorPosition(1)

    @pyqtSlot()
    def setSolver(self):
        """Load caffe solver  for training from solverpath lineedit str

        :return:
        """
        solver_path = self.widgets.network.solver_prototxt_path.text()
        check_result = self.caffe_tools.pathValidationCheck(solver_path, opt='solver')

        self.widgets.network.train_test_prototxt_path.setText(check_result[1])

        if check_result[0]:
            if self.widgets.control.mode.currentText() == 'CPU':
                caffe.set_mode_cpu()

            elif self.widgets.control.mode.currentText() == 'GPU':
                caffe.set_mode_gpu()
                caffe.set_device(0)

            self.solver = caffe.get_solver(solver_path)

            if self.widgets.network.solverstate_import_enable.isChecked():
                solvestate_path = self.widgets.network.sovlerstate_path.text()

                self.log.emit('Solverstate :%s' % solvestate_path, 'Info')
                self.load_solver(solvestate_path)
        else:
            self.log.emit('Load Failed!! Please check file or directory path.', 'Alert')

        # self.solver = caffe.get_solver('model/cifar10_example/solver.prototxt')
        if self.solver is not None:
            self.initMonitoring()

    @pyqtSlot()
    def inoutUpdate(self):
        """input and output layer display
        :param bnum:
        :return:
        """

        bnum = self.widgets.networkmonitor.input_batch_number.value()

        # Update input data
        img = self.input_layer_data[bnum]

        img = np.moveaxis(img, 0, -1)
        img = np.uint8((img - img.min()) / img.ptp() * 255.0)

        height = img.shape[0]
        width = img.shape[1]
        qImg = QImage(width, height, QImage.Format_RGB888)

        for x in range(width):
            for y in range(height):
                qImg.setPixel(y, x, QColor(*img[x, y]).rgb())

        pixmap0 = QPixmap.fromImage(qImg)
        self.widgets.networkmonitor.input_image_display.setPixmap(
            QPixmap(pixmap0).scaled(QSize(100, 100), Qt.KeepAspectRatio))

        # Update output data
        self.res = self.output_layer_data[bnum]
        # self.widgets.networkmonitor.result_topknum.valueChanged.connect(self.topkUpdate)
        self.topkUpdate(self.widgets.networkmonitor.result_topknum.value())

        if img.shape[2] > 1:
            bin_val_red, bin_int_red = np.histogram(img[:, :, 0].flatten(), 50)
            bin_val_green, bin_int_green = np.histogram(img[:, :, 1].flatten(), 50)
            bin_val_blue, bin_int_blue = np.histogram(img[:, :, 2].flatten(), 50)

        self.widgets.networkmonitor.input_redhist.setData(x=bin_int_red[1:], y=bin_val_red, clear=True)
        self.widgets.networkmonitor.input_greenhist.setData(x=bin_int_green[1:], y=bin_val_green, clear=True)
        self.widgets.networkmonitor.input_bluehist.setData(x=bin_int_blue[1:], y=bin_val_blue, clear=True)

    @pyqtSlot()
    def hiddenUpdate(self):
        idx = self.widgets.networkmonitor.input_batch_number.value()
        layer_name = self.widgets.networkmonitor.hidden_layername.currentText()
        hidden_layer_data = self.solver.net.blobs[layer_name].data[idx].copy()
        self.widgets.networkmonitor.hidden_layerdisplay.setTableConfiguration(*calSquare(hidden_layer_data.shape[0], 1))
        self.widgets.networkmonitor.hidden_layerdisplay.setData(hidden_layer_data)

        bin_val, bin_int = np.histogram(hidden_layer_data.flatten(), 50)
        curve = pg.PlotCurveItem(bin_int, bin_val, stepMode=True, fillLevel=0, brush=(20, 115, 155, 80))

        for i in self.widgets.networkmonitor.hidden_histcurve:
            self.widgets.networkmonitor.hidden_histplot.removeItem(i)

        if len(self.widgets.networkmonitor.hidden_histcurve) >= 3:
            self.widgets.networkmonitor.hidden_histcurve.pop(0)
            self.widgets.networkmonitor.hidden_histcurve.append(curve)
        else:
            self.widgets.networkmonitor.hidden_histcurve.append(curve)

        for i in self.widgets.networkmonitor.hidden_histcurve:
            self.widgets.networkmonitor.hidden_histplot.addItem(i)

    @pyqtSlot(int)
    def topkUpdate(self, k):
        res_softmax = np.exp(self.res) / np.exp(self.res).sum()
        res_dict = topk(res_softmax, list(np.arange(len(res_softmax))), k)

        y = list(list(zip(*res_dict))[0])
        x = np.array(self.label)[list(list(zip(*res_dict))[1])].tolist()

        self.widgets.networkmonitor.result_display.setData(x, y)

    def initMonitoring(self):

        ###--- net props initialization ---###
        self.widgets.solvermonitor.net_properties.setRowCount(len(list(self.solver.net._layer_names)))
        for idx, layer in enumerate(self.solver.net._layer_names):
            self.widgets.solvermonitor.net_properties.setItem(idx, 0, QTableWidgetItem(layer))

            try:
                blob_shape = str(list(self.solver.net.blobs[layer].shape))
            except:
                blob_shape = '-'
            self.widgets.solvermonitor.net_properties.setItem(idx, 1, QTableWidgetItem(blob_shape))

            try:
                param_shape = str(list(self.solver.net.params[layer][0].data.shape)) + str(
                    list(self.solver.net.params[layer][1].data.shape))
            except:
                param_shape = '-'
            self.widgets.solvermonitor.net_properties.setItem(idx, 2, QTableWidgetItem(param_shape))

        ###--- Initialization for Solver Performance Visualization Section  ---###
        self.widgets.solvermonitor.forward_time.setData(clear=True)
        self.widgets.solvermonitor.backward_time.setData(clear=True)
        self.widgets.solvermonitor.total_time.setData(clear=True)

        self.widgets.solvermonitor.test_acc_plot.setData(clear=True)
        self.widgets.solvermonitor.train_acc_plot.setData(clear=True)
        self.widgets.solvermonitor.train_loss_plot.setData(clear=True)

        # Network performance visdata dict
        self.time_data_object = {'Iteration': [], 'Forward': [], 'Backward': [], 'Total': []}
        self.train_perf_data_object = {'Iteration': [], 'train_acc': [], 'train_loss': []}
        self.test_perf_data_object = {'Iteration': [], 'test_acc': []}

        ###--- Initialization for Detailed Visualization Section ---###
        # Initial data setting
        self.setMonitoringdata()

        # Input data
        blob_list = list(self.solver.net.blobs.keys())
        num_of_ib = self.solver.net.blobs[blob_list[0]].shape[0]

        self.widgets.networkmonitor.input_batch_number.setMaximum(num_of_ib)
        self.widgets.networkmonitor.input_batch_number.setMinimum(0)
        # self.widgets.networkmonitor.input_batch_number.valueChanged.connect(self.inoutUpdate)
        # self.widgets.networkmonitor.input_batch_number.valueChanged.connect(self.hiddenUpdate)
        self.widgets.networkmonitor.input_batch_number.setValue(0)

        # Output data
        num_of_ob = self.solver.net.blobs[blob_list[-2]].data.shape[-1]

        self.widgets.networkmonitor.result_topknum.setMaximum(num_of_ob)
        self.widgets.networkmonitor.result_topknum.setMinimum(1)
        self.widgets.networkmonitor.result_topknum.setValue(np.ceil(num_of_ob / 10))

        label_file_path = self.widgets.networkmonitor.result_labelpath.text()

        # ::TODO : Sould be changed to Qmessage box
        if label_file_path != '':
            with open(label_file_path, 'r') as flabel:
                self.label = flabel.readlines()

            self.log.emit('Open Label file : %s' % label_file_path, 'info')

        else:
            temp_lb = np.arange(num_of_ob)
            self.label = temp_lb.astype(np.str)

            self.log.emit('Can\'t Open label file : %s' % label_file_path, 'Alert')
            self.log.emit('Serialized counter generated!!', 'Alert')

        self.test_profiler.setLabel(self.label)
        # ::END

        # Hidden Data
        param_list = list(self.solver.net.params.keys())
        self.widgets.networkmonitor.hidden_layername.clear()
        self.widgets.networkmonitor.hidden_layername.addItems(param_list)
        # self.widgets.networkmonitor.hidden_layername.currentTextChanged.connect(self.hiddenUpdate)

        self.inoutUpdate()
        self.hiddenUpdate()

        self.monwidget = {'a': self.widgets.networkmonitor.input_batch_number,
                          'b': self.widgets.networkmonitor.hidden_layername,
                          'c': self.widgets.networkmonitor.result_topknum}

        self.widgets.networkmonitor.input_batch_number.valueChanged.connect(
            lambda z, v=self.monwidget: self.test_profiler.netMonitoring(**v))
        self.widgets.networkmonitor.hidden_layername.currentTextChanged.connect(
            lambda z, v=self.monwidget: self.test_profiler.netMonitoring(**v))
        self.widgets.networkmonitor.result_topknum.valueChanged.connect(
            lambda z, v=self.monwidget: self.test_profiler.netMonitoring(**v))

    def save_solver(self):
        if self.solver is not None:
            self.solver.snapshot()
            self.log.emit('Save Done [iter :%d] Solverstate and Caffemodel' % self.solver.iter, 'Normal')

    def load_solver(self, fname):
        self.solver.restore(fname)
        self.log.emit('Restore Done', 'Info')

    def setMonitoringdata(self):
        self.input_layer_data = self.solver.net.blobs[list(self.solver.net.blobs.keys())[0]].data.copy()
        self.output_layer_data = self.solver.net.blobs[list(self.solver.net.params.keys())[-1]].data.copy()


class trainProfiler(QObject):
    # Signals
    log = pyqtSignal(str, str)
    time_tbl_update = pyqtSignal(int, int, str)
    time_graph_update = pyqtSignal(tuple)

    perf_graph_update = pyqtSignal(tuple)

    input_image_update = pyqtSignal(QPixmap)
    input_red_hist_update = pyqtSignal(dict)
    input_green_hist_update = pyqtSignal(dict)
    input_blue_hist_update = pyqtSignal(dict)

    hidden_hist_update = pyqtSignal(pg.PlotCurveItem)

    mon_update = pyqtSignal()

    output_barplot = pyqtSignal(dict)

    def __init__(self, hidden_table, parent=None):
        super(self.__class__, self).__init__(parent)
        self.solver = None
        self.hidden_table = hidden_table
        self.label = None

    # Slot1. net time update
    @pyqtSlot(object, dict)
    def netUpdate(self, solver, time_data):
        self.log.emit('Time Data Update %s' % list(time_data.keys()), 'Alert')

        self.solver = solver

        # Time Update
        total_forward_time = 0
        total_backward_time = 0
        for idx, layer in enumerate(list(time_data.keys())):
            try:
                forward_time = str('%.3f' % time_data[layer][0])
                backward_time = str('%.3f' % time_data[layer][1])
                total_forward_time += time_data[layer][0]
                total_backward_time += time_data[layer][1]
            except:
                forward_time = '-'
                backward_time = '-'

            self.time_tbl_update.emit(idx, 3, forward_time)
            self.time_tbl_update.emit(idx, 4, backward_time)

        time_data_packet = (solver.iter, total_forward_time, total_backward_time)
        self.time_graph_update.emit(time_data_packet)

        perf_data_packet = (solver.iter, float(solver.net.blobs['loss'].data), np.mean(
            solver.net.blobs[list(solver.net.params.keys())[-1]].data.argmax(1) == solver.net.blobs['label'].data))
        self.perf_graph_update.emit(perf_data_packet)

        self.mon_update.emit()

    # Slot2. net mon update
    @pyqtSlot()
    def netMonitoring(self, a=None, b=None, c=None):
        """
        monwidget = {'a' : self.widgets.networkmonitor.input_batch_number,
             'b' : self.widgets.networkmonitor.hidden_layername,
             'c' : self.widgets.networkmonitor.result_topknum}
        :param bnum:
        :param layer_name:
        :param tnum:
        :return:
        """

        bnum = a.value()
        layer_name = b.currentText()
        tnum = c.value()

        if self.solver is not None:
            input_layer_data = self.solver.net.blobs[list(self.solver.net.blobs.keys())[0]].data[bnum].copy()
            hidden_layer_data = self.solver.net.blobs[layer_name].data[bnum].copy()
            output_layer_data = self.solver.net.blobs[list(self.solver.net.params.keys())[-1]].data[bnum].copy()

            self.log.emit('Data Setting ::input %s:: hidden %s :: output :: %s ' % (input_layer_data.shape,
                                                                                    hidden_layer_data.shape,
                                                                                    output_layer_data.shape), 'Notify')

            # :: Input Image Generation ::
            img = np.moveaxis(input_layer_data, 0, -1)
            img = np.uint8((img - img.min()) / img.ptp() * 255.0)

            height = img.shape[0]
            width = img.shape[1]
            qImg = QImage(width, height, QImage.Format_RGB888)

            for x in range(width):
                for y in range(height):
                    qImg.setPixel(y, x, QColor(*img[x, y]).rgb())

            pixmap0 = QPixmap.fromImage(qImg)

            self.input_image_update.emit(QPixmap(pixmap0).scaled(QSize(100, 100), Qt.KeepAspectRatio))

            # :: Input Histogram Generation ::
            temp_bin = np.arange(0, 256, 5)
            if img.shape[2] > 1:
                bin_val_red, bin_int_red = np.histogram(img[:, :, 0].flatten(), temp_bin)
                bin_val_green, bin_int_green = np.histogram(img[:, :, 1].flatten(), temp_bin)
                bin_val_blue, bin_int_blue = np.histogram(img[:, :, 2].flatten(), temp_bin)

            self.input_red_hist_update.emit({'x': bin_int_red[1:], 'y': bin_val_red, 'clear': True})
            self.input_green_hist_update.emit({'x': bin_int_green[1:], 'y': bin_val_green, 'clear': True})
            self.input_blue_hist_update.emit({'x': bin_int_blue[1:], 'y': bin_val_blue, 'clear': True})

            # :: Weight Image Generation

            self.hidden_table.setTableConfiguration(*calSquare(hidden_layer_data.shape[0], 1))
            self.hidden_table.setData(hidden_layer_data)

            bin_val, bin_int = np.histogram(hidden_layer_data.flatten(), 50)
            curve = pg.PlotCurveItem(bin_int, bin_val, stepMode=True, fillLevel=0, brush=(20, 115, 155, 80))

            self.hidden_hist_update.emit(curve)

            # :: Output Prob Barplot Generation
            res_softmax = np.exp(output_layer_data) / np.exp(output_layer_data).sum()
            res_dict = topk(res_softmax, list(np.arange(len(res_softmax))), tnum)

            y = list(list(zip(*res_dict))[0])
            x = np.array(self.label)[list(list(zip(*res_dict))[1])].tolist()

            self.output_barplot.emit({'x': x, 'y': y})

        else:
            self.log.emit('Solver is not Loading Yet...', 'Alert')

    def setLabel(self, label):
        self.label = label

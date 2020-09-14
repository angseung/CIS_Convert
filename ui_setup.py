"""
.. module:: ui_setup.py
   :platform: Unix, Windows
   :synopsis: Load PLATFORM6573_v1 UI file and setup.
.. moduleauthor:: Seokhun Jeon <seokhun.jeon@keti.re.kr>
"""

from PyQt5 import uic
import icon_rc
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDialog, QTableWidgetItem, QDockWidget
from proto_generator import neural_net_generator
import sys

import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOptions(antialias=True)

# CAFFE import TODO: shoud be changed dialog routiune!!!
try:
    import caffe
    from caffe.proto import caffe_pb2

except ImportError:
    try:
        with open('caffe_path.txt', 'r') as f:
            caffe_path = f.read()
        sys.path.append(caffe_path)

    except FileNotFoundError:
        dialog = QFileDialog(None, 'Caffe Path')
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_() == QDialog.Accepted:
            caffe_path = dialog.selectedFiles()[0]
            with open('caffe_path.txt', 'w') as f:
                f.write(caffe_path)

            sys.path.append(caffe_path)

        try:
            import caffe
            from caffe.proto import caffe_pb2

        except Exception as e:
            print('Error: ', e)
            sys.exit()

from tools import tableWidget, getFilePath, structure, caffeTools, vbarGraph
from ui_control import logMessage, trainingMain, testMain


class form(QMainWindow):
    """This is the UI form for DNN Simulation.
    .. note::
       Default setting and event connection.
    """

    def __init__(self):
        """ UI constructor
        """
        super(form, self).__init__()

        # Load UI file
        uic.loadUi('nnp_main_r2.ui', self)
        self.initWidgets()
        self.defineWidgetlist()

        # Check Caffe
        self.caffetools = caffeTools()

        # log window
        self.log_window = logMessage(self.log_widgets)

        # training main
        self.tm = trainingMain(self.training_widgets)
        self.tm.log.connect(self.log_window.setLog)

        # test main
        self.tstm = testMain(self.test_widgets)
        self.tstm.log.connect(self.log_window.setLog)

        # neural_net dock module
        self.nng = neural_net_generator(self.tableWidget_5, self.plainTextEdit_4)
        self.nng.log.connect(self.log_window.setLog)

        self.signalConnection()
        self.setDefaultValues()

        self.actionOpen_Prototxt.triggered.connect(self.nng.open_network)
        self.toolButton_7.clicked.connect(self.nng.add_layer)
        self.toolButton_6.clicked.connect(self.nng.remove_layer)
        self.toolButton.clicked.connect(self.nng.move_down_layer)
        self.toolButton_8.clicked.connect(self.nng.move_up_layer)
        self.toolButton_2.clicked.connect(self.nng.reset_network)
        self.pushButton_4.clicked.connect(self.nng.save_network)

    def initWidgets(self):
        ###---- Training tab monitoring initialization ----###
        # Netproperties widgtets
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)

        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)

        column_headers = ['layers', 'blob_size', 'param_size', 'forward', 'backward']
        self.tableWidget.setColumnCount(len(column_headers))
        self.tableWidget.setHorizontalHeaderLabels(column_headers)
        self.tableWidget.resizeColumnsToContents()

        # Process time widgets
        self.widget_16.setLabel('left', 'Time', units='ms')
        self.widget_16.setLabel('bottom', 'Iteration', units='x1000')
        self.widget_16.showGrid(x=False, y=True)

        # Performance measurement widgets
        self.p1 = self.widget_22.plotItem
        self.p1.setLabels(left='Accuracy')
        self.p1.setLabels(bottom='Iteration')

        self.p2 = pg.ViewBox()
        self.p1.showAxis('right')
        self.p1.scene().addItem(self.p2)
        self.p1.getAxis('right').linkToView(self.p2)
        self.p2.setXLink(self.p1)
        self.p1.getAxis('right').setLabel('Loss', color='#0000ff')

        self.train_loss_plot_item = pg.PlotCurveItem(pen='r')
        self.p2.addItem(self.train_loss_plot_item)
        self.p1.vb.sigResized.connect(self.__update_views)

    def defineWidgetlist(self):
        ###---- Training tab widgets ----###
        self.training_widgets = structure()

        # Network widget group
        self.training_widgets.network = structure()

        self.training_widgets.network.solver_prototxt_path = self.lineEdit_3
        self.training_widgets.network.train_test_prototxt_path = self.lineEdit_5
        self.training_widgets.network.sovlerstate_path = self.lineEdit_6

        self.training_widgets.network.solverstate_import_enable = self.checkBox_4

        self.training_widgets.network.solver_prototxt_filedialog = self.toolButton_12
        self.training_widgets.network.solverstate_filedialog = self.toolButton_15

        self.training_widgets.network.load_button = self.pushButton_2

        # Control widget group
        self.training_widgets.control = structure()

        self.training_widgets.control.progressbar = self.progressBar
        self.training_widgets.control.iteration = self.lineEdit
        self.training_widgets.control.throughput = self.lineEdit_2
        self.training_widgets.control.mode = self.comboBox_2

        self.training_widgets.control.run_button = self.pushButton
        self.training_widgets.control.savesolver_button = self.pushButton_8

        # Solver monitoring widget group
        self.training_widgets.solvermonitor = structure()

        self.training_widgets.solvermonitor.net_properties = self.tableWidget
        self.training_widgets.solvermonitor.forward_time = self.widget_16.plot(pen='g')
        self.training_widgets.solvermonitor.backward_time = self.widget_16.plot(pen='b')
        self.training_widgets.solvermonitor.total_time = self.widget_16.plot(pen='r')

        self.training_widgets.solvermonitor.test_acc_plot = self.p1.plot(pen='b')
        self.training_widgets.solvermonitor.train_acc_plot = self.p1.plot(pen='g')
        self.training_widgets.solvermonitor.train_loss_plot = self.train_loss_plot_item

        # Network Layer monitoring widget group
        self.training_widgets.networkmonitor = structure()

        self.training_widgets.networkmonitor.input_batch_number = self.spinBox
        self.training_widgets.networkmonitor.input_image_display = self.label_19
        self.training_widgets.networkmonitor.input_histogram_enable = self.toolButton_9
        self.training_widgets.networkmonitor.input_histplot = self.widget_36
        self.training_widgets.networkmonitor.input_redhist = self.widget_36.plot(pen='r')
        self.training_widgets.networkmonitor.input_greenhist = self.widget_36.plot(pen='g')
        self.training_widgets.networkmonitor.input_bluehist = self.widget_36.plot(pen='b')

        self.training_widgets.networkmonitor.hidden_layername = self.comboBox
        self.training_widgets.networkmonitor.hidden_layerdisplay = tableWidget(0, 0,
                                                                               self.widget_11)  # promoted to custom tablewidget
        self.training_widgets.networkmonitor.hidden_histogram_enable = self.toolButton_10
        self.training_widgets.networkmonitor.hidden_histcurve = []
        self.training_widgets.networkmonitor.hidden_histplot = self.widget_9

        self.training_widgets.networkmonitor.result_topknum = self.spinBox_3
        self.training_widgets.networkmonitor.result_display = vbarGraph(self.widget_12)  # promoted to pyqtgraph
        self.training_widgets.networkmonitor.result_labelpath = self.lineEdit_9
        self.training_widgets.networkmonitor.label_filedialog = self.toolButton_11

        self.test_widgets = structure()

        self.test_widgets.netinputconfig = structure()

        self.test_widgets.netinputconfig.deploy_path = self.lineEdit_7
        self.test_widgets.netinputconfig.caffemodel_path = self.lineEdit_8
        self.test_widgets.netinputconfig.load_button = self.pushButton_5
        self.test_widgets.netinputconfig.input_mode = self.comboBox_6
        self.test_widgets.netinputconfig.input_type = [self.radioButton_3, self.radioButton_4]
        self.test_widgets.netinputconfig.inputfile_path = self.lineEdit_4
        self.test_widgets.netinputconfig.intput_cam = self.comboBox_7
        self.test_widgets.netinputconfig.cam_resolution = self.comboBox_3
        self.test_widgets.netinputconfig.cam_rescale = self.spinBox_5

        self.test_widgets.test_configs = structure()
        self.test_widgets.test_configs.run_test = self.pushButton_7
        self.test_widgets.test_configs.test_mode = self.toolButton_17
        self.test_widgets.test_configs.disp_label = self.lineEdit_10
        self.test_widgets.test_configs.mode = self.comboBox_4

        self.test_widgets.networkmonitor = structure()

        self.test_widgets.networkmonitor.input_image_display = self.label_24
        self.test_widgets.networkmonitor.layer_name = self.comboBox_9
        self.test_widgets.networkmonitor.layer_type = self.comboBox_8
        self.test_widgets.networkmonitor.hidden_dsiplay = tableWidget(0, 0,self.widget_32)

        self.test_widgets.networkmonitor.layer_channel = self.spinBox_8

        self.log_widgets = structure()
        self.log_widgets.log = self.plainTextEdit
        self.log_widgets.verbose_button = self.toolButton_3
        self.log_widgets.logsave_button = self.toolButton_4
        self.log_widgets.main_dock = self.dockWidget

    def signalConnection(self):
        self.training_widgets.network.solver_prototxt_filedialog.clicked.connect \
            (lambda _: getFilePath(self.training_widgets.network.solver_prototxt_path, 'SolverPrototxt File'))
        self.training_widgets.network.solverstate_filedialog.clicked.connect \
            (lambda _: getFilePath(self.training_widgets.network.sovlerstate_path, 'SolverState File'))
        self.training_widgets.networkmonitor.label_filedialog.clicked.connect \
            (lambda _: getFilePath(self.training_widgets.networkmonitor.result_labelpath, 'Label Path'))

        self.toolButton_16.clicked.connect(
            lambda _: getFilePath(self.test_widgets.netinputconfig.deploy_path, 'Deployprototxt File'))
        self.toolButton_20.clicked.connect(
            lambda _: getFilePath(self.test_widgets.netinputconfig.caffemodel_path, 'Caffemodel File'))
        self.toolButton_18.clicked.connect(
            lambda _: getFilePath(self.test_widgets.netinputconfig.inputfile_path, 'Input Image(Video) File'))

        self.netconfig_dock.toggleViewAction().toggled.connect(self.actionNetConfigs.setChecked)
        self.dockWidget.toggleViewAction().toggled.connect(self.actionlogmessage.setChecked)

        self.training_widgets.network.load_button.clicked.connect(self.tm.setSolver)

    def __update_views(self):
        self.p2.setGeometry(self.p1.vb.sceneBoundingRect())
        self.p2.linkedViewChanged(self.p1.vb, self.p2.XAxis)

    def setDefaultValues(self):
        self.training_widgets.control.throughput.setEnabled(False)
        self.training_widgets.control.iteration.setText('1000')  # for testing
        self.widget_36.setVisible(False)
        self.widget_30.setVisible(False)
        self.widget_9.setVisible(False)
        self.label_44.setVisible(False)

        # TODO : This is TEST MODE CODE!! ELIMINATION LATER!!
        self.lineEdit_7.setText('model/YoloV2/yoloV2Tiny20.prototxt')
        self.lineEdit_8.setText('model/YoloV2/yoloV2Tiny20.caffemodel')

        self.actionNetConfigs.toggled.connect(self.netconfig_dock.setVisible)
        self.actionlogmessage.toggled.connect(self.dockWidget.setVisible)

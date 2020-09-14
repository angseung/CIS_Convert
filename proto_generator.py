from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QDialog, QWidget, QPushButton, QFileDialog, QListWidgetItem, QVBoxLayout
from PyQt5 import uic

from google.protobuf import text_format
import os, sys

#sys.path.append('C:/caffe/caffe/python')
#sys.path.append('D:/caffe/python')

from caffe.proto import caffe_pb2
from collections import OrderedDict
from tools import getFilePath

node_list = OrderedDict()

node_list['Spatial, Image Nodes'] = ['Pooling Node',
                                     'Convolution Node',
                                     'DeConvolution Node',
                                     'LRN Node',
                                     'Concatanation Node',
                                     'Slice Node',
                                     'BatchNorm Node']
node_list['Neuron, Elementwise Nodes'] = ['MVN Node',
                                          'Exponential Node',
                                          'Elementwise Node',
                                          'ArgMax Node',
                                          'InnerProduct Node',
                                          'Flatten Node',
                                          'Activation Node',
                                          'ReLU Node',
                                          'PReLU Node',
                                          'Dropout Node',
                                          'Log Node',
                                          'Power Node']
node_list['Loss Nodes'] = ['Accuracy Node',
                           'EULoss Node',
                           'SCELoss Node',
                           'SoftmaxWithLoss Node',
                           'Reduction Node',
                           'PythonLoss Node']
node_list['Solver, Data Nodes'] = ['Solver Node',
                                   'Data Node',
                                   'HDF5Output Node']
node_list['Misc'] = ['Silence Node']

layer_type = {'Data Node': 'Input',
              'Pooling Node': 'Pooling',
              'Convolution Node': 'Convolution'
              }


class neural_net_generator(QObject):
    log = pyqtSignal(str, str)
    status = pyqtSignal(str)

    bidx = -5

    def __init__(self, layer_list, dock):
        super().__init__()

        self.layer_list = layer_list
        self.dock = dock

        self.reset_network()

    def reset_network(self):
        self.net = None
        self.network_model = []
        self.layer_list.clear()

    def open_network(self):
        self.status.emit('Open Neural Network!!')
        self.reset_network()

        self.otm = open_test_model_dialog()

        self.otm.pushButton.clicked.connect(self.__caffe_protobuf_parser)
        self.otm.show()
        self.otm.exec_()

    def __caffe_protobuf_parser(self):
        self.otm.close()

        if self.otm.get_mode() == 'TEST':  # Test Mode
            self.net = caffe_pb2.NetParameter()

            with open(self.otm.lineEdit_2.text(), 'r') as f:
                text_format.Merge(f.read(), self.net)

            self.layer_list.setRowCount(len(self.net.layer))
            self.layer_list.setColumnCount(1)

            for idx, clayer in enumerate(self.net.layer):
                layer_button = QPushButton(clayer.name)
                self.layer_list.setCellWidget(idx, 0, layer_button)

                layer_button.clicked.connect(lambda state, x=idx: self.__update_layer_props(x))
            self.layer_list.resizeColumnsToContents()
            self.layer_list.setColumnWidth(0, 120)

        else:  # Training Mode
            self.solver = caffe_pb2.SolverParameter()

            with open(self.otm.lineEdit.text(), 'r') as f:
                text_format.Merge(f.read(), self.solver)

            self.solver_org = self.solver.__deepcopy__()

            self.net = caffe_pb2.NetParameter()

            with open(self.solver.net, 'r') as f:
                text_format.Merge(f.read(), self.net)

            self.net_org = self.net.__deepcopy__()

            self.layer_list.setRowCount(len(self.net.layer) + 1)
            self.layer_list.setColumnCount(1)

            for idx, clayer in enumerate(self.net.layer):
                layer_button = QPushButton(clayer.name)
                self.layer_list.setCellWidget(idx, 0, layer_button)

                layer_button.clicked.connect(lambda state, x=idx: self.__update_layer_props(x))

            layer_button = QPushButton('Solver')
            self.layer_list.setCellWidget(idx + 1, 0, layer_button)
            layer_button.clicked.connect(lambda state, x=-1: self.__update_layer_props(x))

            self.layer_list.resizeColumnsToContents()
            self.layer_list.setColumnWidth(0, 120)

    def __update_layer_props(self, idx):
        self.layer_config = self.dock.toPlainText()
        #print(self.layer_config, self.bidx, idx)
        self.dock.clear()

        if self.bidx > 0:
            self.net.layer[self.bidx].Clear()
            text_format.Merge(self.layer_config, self.net.layer[self.bidx])
        elif self.bidx == -1:
            self.solver.Clear()
            text_format.Merge(self.layer_config, self.solver)

        if idx == -1:
            self.dock.insertPlainText(str(self.solver))
        else:
            self.dock.insertPlainText(str(self.net.layer[idx]))

        self.bidx = idx

    def save_network(self):
        # Check layer path
        print('Save Network')

        files_types = "All Files (*);;Prototxt Files (*.prototxt)"

        net_file_name, _ = QFileDialog.getSaveFileName(QWidget(), 'Save Net Prototxt', '', files_types)

        if net_file_name != '':
            with open(net_file_name, 'w') as f:
                f.write(text_format.MessageToString(self.net))

            self.log.emit('Save Net Prototxt : %s' % net_file_name, 'Info')
            self.solver.net = net_file_name

        else:
            self.log.emit('Cancel Save Net Prototxt', 'Notify')

        solver_file_name, _ = QFileDialog.getSaveFileName(QWidget(), 'Save Solver Prototxt', '', files_types)

        if solver_file_name != '':
            with open(solver_file_name, 'w') as f:
                f.write(text_format.MessageToString(self.solver))
            self.log.emit('Save Solver Prototxt : %s' % solver_file_name, 'Info')
        else:
            self.log.emit('Cancel Save Solver Prototxt', 'Notify')

    def add_layer(self):
        self.ald = add_layer_dialog()

        self.ald.pushButton.clicked.connect(self._update_list)

        self.ald.show()
        self.ald.exec_()

    def remove_layer(self):
        pass

    def move_up_layer(self):
        pass

    def move_down_layer(self):
        pass

    def __update_network_list(self, layer):

        layer_obj = layerWidget(layer.name)
        self.network_model.append(layer_obj)

        w = layer_obj.get_item(self.layer_list)
        b = layer_obj.get_widget()
        layer_obj.widget_button.clicked.connect(self.__update_layer_props)

        self.layer_list.addItem(w)
        self.layer_list.setItemWidget(w, b)

    def __solver_parser(self, path):
        solver = caffe_pb2.SolverParameter()

        with open(path, 'r') as f:
            text_format.Merge(f.read(), solver)

        return solver

    def __net_parser(self, path):
        net = caffe_pb2.NetParameter()

        with open(path, 'r') as f:
            text_format.Merge(f.read(), net)

        return net

    def _update_list(self):
        self.ald.close()
        print(self.ald.comboBox_2.currentText())
        # self.network_model.append(layerWidget(self.ald.comboBox_2.currentText()))
        if self.net is None:
            self.net = caffe_pb2.NetParameter()
            self.layer_list.setColumnCount(1)

        self.net.layer.add()

        text_format.Merge('type: "%s"' % layer_type[self.ald.comboBox_2.currentText()], self.net.layer[-1])

        self.layer_list.setRowCount(len(self.net.layer))

        layer_button = QPushButton(self.ald.comboBox_2.currentText())
        print(len(self.net.layer) - 1)
        self.layer_list.setCellWidget(len(self.net.layer) - 1, 0, layer_button)

        layer_button.clicked.connect(lambda state, x=len(self.net.layer) - 1: self.__update_layer_props(x))
        self.layer_list.resizeColumnsToContents()
        self.layer_list.setColumnWidth(0, 120)

    def _show_props(self):
        idx = self.layer_list.selectedIndexes()[0].row()

        print(idx)
        for i in reversed(range(self.dock.layout().count())):
            self.dock.layout.itemAt(i).widget().setParent(None)

        self.dock.layout().addWidget(QPushButton('Test!!!'))


class add_layer_dialog(QDialog):
    def __init__(self):
        super(add_layer_dialog, self).__init__()

        uic.loadUi('add_layer_dialog.ui', self)
        self.init_list = list(node_list.keys())
        self.comboBox.addItems(self.init_list)
        self.comboBox_2.addItems(node_list[self.init_list[0]])

        self.comboBox.currentIndexChanged.connect(self.update_layer_type)

    def update_layer_type(self):
        self.comboBox_2.clear()
        self.comboBox_2.addItems(node_list[self.init_list[self.comboBox.currentIndex()]])


class layerWidget(QObject):
    def __init__(self, name):
        super().__init__()
        self.widget = QWidget()
        self.widget_button = QPushButton(name)
        self.widget.setLayout(QVBoxLayout())

        self.widget.layout().addWidget(self.widget_button)
        # self.widget.layout().addStretch()

    def get_item(self, listwidget):
        itemN = QListWidgetItem(listwidget)
        itemN.setSizeHint(self.widget.sizeHint())

        return itemN

    def get_widget(self):
        return self.widget

    def prop_widget(self):
        widget = QWidget()

        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QPushButton('Test'))

        return widget


class open_test_model_dialog(QDialog):
    deploy_file_path = None
    caffemodel_file_path = None

    mode = 'TRAIN'

    def __init__(self):
        super(open_test_model_dialog, self).__init__()

        uic.loadUi('proto_open_dialog.ui', self)

        self.toolButton.clicked.connect(lambda _: getFilePath(self.lineEdit, 'Open Train_Valid Prototxt'))

        self.toolButton_2.clicked.connect(lambda _: getFilePath(self.lineEdit_2, 'Open Deploy Prototxt'))
        self.toolButton_3.clicked.connect(lambda _: getFilePath(self.lineEdit_3, 'Open Caffemodel Prototxt'))

        self.radioButton.clicked.connect(self.slot_clicked_item)
        self.radioButton_2.clicked.connect(self.slot_clicked_item)

    def slot_clicked_item(self):
        if self.radioButton.isChecked():
            self.stackedWidget.setCurrentIndex(0)
            self.mode = 'TRAIN'
        else:
            self.stackedWidget.setCurrentIndex(1)
            self.mode = 'TEST'

    def get_mode(self):
        return self.mode

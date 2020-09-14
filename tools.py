"""
.. module:: tools.py
   :platform: Unix, Windows
   :synopsis: custom widgets and common custom classes
.. moduleauthor:: Seokhun Jeon <seokhun.jeon@keti.re.kr>
"""

from PyQt5.QtWidgets import QFileDialog, QTableWidget, QVBoxLayout, QWidget, QLabel, QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt, QPoint, QSize, QObject

from google.protobuf import text_format
import numpy as np
import sys, os
import pyqtgraph as pg

from caffe.proto import caffe_pb2

def getFilePath(displayWidget, title):
    """getFilePath
    :param displayWidget: path indicator (line edit)
    :param title: file dialog title
    :return:
    """

    path = QFileDialog.getOpenFileName(None, title)

    if path !='':
        displayWidget.setText(str(path[0]))

class tableWidget(QTableWidget):
    def __init__(self, row, col, parent=None):
        QTableWidget.__init__(self)

        parent.setLayout(QVBoxLayout())
        parent.layout().addWidget(self)
        self.setTableConfiguration(row, col)
        #self.setMouseTracking(True)

        self.cellClicked.connect(self.onCellClicked)

    def setTableConfiguration(self, row, col):
        self.setColumnCount(col)
        self.setRowCount(row)

        self.row = row
        self.col = col

        self.lb = ['' for i in range(row * col)]

        self.horizontalHeader().hide()
        self.verticalHeader().hide()

        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    @pyqtSlot()
    def setData(self, weight):
        self.clearContents()
        print(self.row,self.col)
        for idx, val in enumerate(weight):
            self.lb[idx] = resSegment(weight[idx])
            self.setCellWidget(int(idx/self.col), idx%self.col, self.lb[idx])
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    @pyqtSlot(int, int)
    def onCellClicked(self, row, column):
        print('Will be added!!')
        '''
        a = QDialog()
        idx = row * self.col + column
        a.setContentsMargins(0,0,0,0)
        a.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        a.setObjectName('test')
        a.setLayout(QVBoxLayout())
        self.lb[idx].setResize(QSize(100,100))
        a.layout().addWidget(self.lb[idx])
        point = self.rect().bottomRight()
        a.move(point - QPoint(self.x_cord, self.y_cord))
        self.setCellWidget(row, column, self.lb[idx])
        a.exec_()


        print(row,column, idx)
        '''
    #def mouseMoveEvent(self, event):
    #    self.x_cord = event.x()
    #    self.y_cord = event.y()

class resSegment(QWidget):
    def __init__(self, img, parent=None):
        QWidget.__init__(self, parent)

        self._img = img

        self.setLayout(QVBoxLayout())
        self.lbPixmap = QLabel(self)

        self.lbPixmap.setStyleSheet("QLabel {background-color: white;padding: 0px 0px 0px 0px;}")

        self.layout().addWidget(self.lbPixmap)

        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.initUi()

    def setBackgroundColor(self, color):
        self.lbPixmap.setStyleSheet("QLabel {background-color: %s;padding: 0px 0px 0px 0px;}" % color)

    def initUi(self):
        qImg = imageReconstuction(self._img, 'gray')

        self.pixmap0 = QPixmap.fromImage(qImg)
        self.lbPixmap.setPixmap(QPixmap(self.pixmap0).scaled(self.lbPixmap.size(), Qt.KeepAspectRatio))

    def setResize(self, size):
        self.lbPixmap.setPixmap(QPixmap(self.pixmap0).scaled(size, Qt.KeepAspectRatio))


def imageReconstuction(input, sub_mode='gray'):
    # input arguments [3] ====
    # input : 2D or 3D image data
    # dispMode : time, image option
    # sub_mode : gray, color mode for time series data reconstuction
    # output [1] ===
    # qImg : output image array for qpixmap display

    input = np.uint8((input - input.min()) / input.ptp() * 255.0)

    height = input.shape[0]
    width = input.shape[1]
    #height = int(np.sqrt(len(input)))
    #width = int(len(input) // height)

    if sub_mode == 'gray':
        # height, width =  self._img.shape
        qImg = QImage(input.data, width, height, width, QImage.Format_Indexed8)
    else:
        qImg = QImage(input.data, width, height, QImage.Format_RGB888)

    return qImg

def topk(elements, labels, k=5):
    top = elements.argsort()[-k:][::-1]
    probs = elements[top]
    return list(zip(probs, np.array(labels)[top]))

class vbarGraph(object):
    def __init__(self, widget):
        object.__init__(self)

        self.axis_info = pg.AxisItem(orientation='left')
        pw = pg.PlotWidget(axisItems={'left': self.axis_info})
        pw.setLabel('left','Top-k')
        pw.setLabel('bottom','Accuracy', units = '%')

        pw.showGrid(x=True, y=True)
        self.output_bargraph = pg.BarGraphItem(x=[], height=[], width=0.9, brush='r')
        self.output_bargraph.rotate(90)
        pw.setXRange(0, 1.2, padding=0)

        pw.addItem(self.output_bargraph)

        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(pw)

    def setData(self, x, y):
        xdict = dict(enumerate(x))
        self.axis_info.setTicks([xdict.items()])
        self.output_bargraph.setOpts(x=list(xdict.keys()), height=-1 * np.array(y))

class caffeTools(QObject):
    log = pyqtSignal(str, str)


    def __init__(self):
        super().__init__()
        pass

    def pathValidationCheck(self, path, opt='net'):
        result = True
        if opt == 'solver':
            self.solver = self.__solver_parser(path)

            if os.path.isfile(self.solver.net):
                self.log.emit('Net path True!', 'Normal')
                self.net = self.__net_parser(self.solver.net)

                for i in self.net.layer:
                    if i.data_param.source != '':
                        if os.path.isdir(i.data_param.source):
                            self.log.emit('Path True %s' % i.data_param.source, 'Normal')
                            result &= True
                        else:
                            self.log.emit('Path Fail %s' % i.data_param.source, 'Alert')
                            result &= False
                    if i.transform_param.mean_file != '':
                        if os.path.isfile(i.transform_param.mean_file):
                            self.log.emit('Path True %s' % i.transform_param.mean_file, 'Normal')
                            result &= True
                        else:
                            self.log.emit('Path Fail %s' % i.transform_param.mean_file, 'Alert')
                            result &= False
            else:
                self.log.emit('Net path False! %s' % self.solver.net, 'Alert')
                result &= False
            return [result, self.solver.net]

        elif opt == 'net':
            self.net = self.__net_parser(path)

            for i in self.net.layer:
                if i.data_param.source != '':
                    if os.path.isdir(i.data_param.source):
                        self.log.emit('Path True %s' % i.data_param.source, 'Normal')
                        result &= True
                    else:
                        self.log.emit('Path Fail %s' % i.data_param.source, 'Alert')
                        result &= False
                if i.transform_param.mean_file != '':
                    if os.path.isfile(i.transform_param.mean_file):
                        self.log.emit('Path True %s' % i.transform_param.mean_file, 'Normal')
                        result &= True
                    else:
                        self.log.emit('Path Fail %s' % i.transform_param.mean_file, 'Alert')
                        result &= False

            return result

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

def calSquare(x, scaler):
    col = int(np.ceil(np.sqrt(x)))
    row = col

    row = int(np.ceil(row*scaler)) + (row%scaler > 0)
    col = int(np.ceil(col/scaler)) + (col%scaler > 0)

    return col, row


class structure(object):
    pass

from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal, pyqtSlot)
import numpy as np
import tempfile, os, sys
import caffe
from collections import OrderedDict
from datetime import datetime

class trainingCore(QThread):
    """This is caffe framwork based dnn sw core

    """
    net_out = pyqtSignal(object, dict)
    training_progress = pyqtSignal(int)
    log = pyqtSignal(str, str)
    val_out = pyqtSignal(tuple)

    def __init__(self, solver = None, iter = 0, mode = 'CPU', test_interval= 100, disp_interval = 10):
        super().__init__()
        self.solver = solver
        self.iter = iter
        self.disp_interval = solver.param.display  # option
        self.test_interval = test_interval

        self.solver_mode = mode

        self.timedata = OrderedDict()
        for i in range(len(self.solver.net.layers)):
            self.timedata[self.solver.net._layer_names[i]] = [0, 0]

        self.finish_flag = False
        self.pause_flag = False

    def __del__(self):
        self.wait()

    def setFinish(self):
        self.finish_flag = True

    def setPause(self, state):
        self.pause_flag = state

    def run(self):
        # For ready to solver thread set-up
        self.sleep(1)

        if self.solver_mode == 'CPU':
            caffe.set_mode_cpu()

        elif self.solver_mode == 'GPU':
            caffe.set_mode_gpu()
            caffe.set_device(0)

        param_list = list(self.solver.net.params.keys())

        self.solver.add_callback(self.caffelog_b_callback, self.caffelog_a_callback)

        self.__caffeTimer()
        caffe_iter_counter = 0
        # Main Training Loop
        # Initial net out emit!!
        self.net_out.emit(self.solver, self.timedata)

        while not self.finish_flag:

            self.solver.step(1)
            self.msleep(10)


            # Update Iteration Status and network result
            caffe_iter_counter += 1
            self.training_progress.emit(int(self.solver.iter / self.iter * 100) + 1)

            if caffe_iter_counter % self.disp_interval == 0:
                self.net_out.emit(self.solver, self.timedata)
                self.__caffeTimer()

            if caffe_iter_counter % self.test_interval == 0:
                pass
                #self.valid_core = validationCore(self.solver.test_nets[0], 100)
                #self.valid_core.log.connect(lambda a,b : self.log.emit(a,b))
                #self.valid_core.val_out.connect(lambda a,b = caffe_iter_counter : self.val_out.emit((b, a)))
                #self.valid_core.start()
                #self.log.emit('Run Validation!!', 'Info')
                #test_acc = []
                #for test_it in range(100):
                #    test_acc.append(self.solver.test_nets[0].forward()['accuracy'])
                #    self.msleep(10)

                #mean_test_acc = np.mean(test_acc)
                #self.val_out.emit((caffe_iter_counter, mean_test_acc))
                #self.log.emit('Iteration %s, Test Accuracy %s ,Test Iteration %s'%(caffe_iter_counter, mean_test_acc, 100), 'Info')

            # Check Control Status
            if self.pause_flag:
                while self.finish_flag: self.msleep(10)

            if caffe_iter_counter == self.iter:
                self.finish_flag = True

    def caffelog_b_callback(self):
        self.tmp =  tempfile.TemporaryFile(mode='w+t')
        os.dup2(self.tmp.fileno(), sys.stdout.fileno())
        os.dup2(self.tmp.fileno(), sys.stderr.fileno())

    def caffelog_a_callback(self):
        self.tmp.seek(0)
        caffelog = self.tmp.readlines()
        if caffelog != '':
            for cmsg in caffelog:
                self.log.emit(cmsg, 'Notify')
        self.tmp.close()

    def __caffeTimer(self):
        fprops = []
        bprops = []

        total = caffe.Timer()
        allrd = caffe.Timer()

        for nlayer in range(len(self.solver.net.layers)):
            fprops.append(caffe.Timer())
            bprops.append(caffe.Timer())

        def show_time():
            if self.solver.iter % (self.solver.param.display) == 0:
                for i in range(len(self.solver.net.layers)):
                    self.timedata[self.solver.net._layer_names[i]] = [fprops[i].ms, bprops[i].ms]

        self.solver.net.before_forward(lambda layer: fprops[layer].start())
        self.solver.net.after_forward(lambda layer: fprops[layer].stop())
        self.solver.net.before_backward(lambda layer: bprops[layer].start())
        self.solver.net.after_backward(lambda layer: bprops[layer].stop())

        self.solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
        self.solver.add_callback(lambda: "", lambda: (allrd.stop(), show_time()))


class validationCore(QThread):
    """This is caffe framwork based dnn sw core

    """
    log = pyqtSignal(str, str)
    val_out = pyqtSignal(float)

    def __init__(self, test_net = None, iter = 0):
        super().__init__()
        self.test_net = test_net
        self.iter = iter

        self.finish_flag = False
        self.pause_flag = False

    def __del__(self):
        self.wait()

    def setFinish(self):
        self.finish_flag = True

    def setPause(self, state):
        self.pause_flag = state

    def run(self):
        self.log.emit('Run Validation!!', 'Info')
        test_acc = []
        for test_it in range(self.iter):
            test_acc.append(self.test_net.forward()['accuracy'])

        mean_test_acc = np.mean(test_acc)
        self.val_out.emit(mean_test_acc)
        self.log.emit('Iteration %s, Test Accuracy %s' % (self.iter, mean_test_acc), 'Info')


class testCore(QThread):
    net_out = pyqtSignal(object)
    log = pyqtSignal(str, str)

    def __init__(self, input, net = None):
        super().__init__()
        self.input = input
        self.net = net

    def __del__(self):
        print(".... End RunCaffe Test Thread ....")
        self.quit()
        self.wait()

    def run(self):

        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))

        stime = datetime.now()

        self.net.forward_all(data=np.asarray([transformer.preprocess('data', self.input_img)]))

        elapsedTime = datetime.now() - stime
        self.log.emit('total time is "%d milliseconds'% elapsedTime.total_seconds() * 1000, 'Notify')

        self.net_out.emit(self.net)
from PyQt5.QtCore import (QObject, QThread, QThreadPool,QRunnable, pyqtSignal, pyqtSlot)

from multiprocessing import Process, Pool

import caffe
import os
import time

class SolverWithCallback(QObject):
    validation_data = pyqtSignal(object)


    def __init__(self):
        super().__init__()

        self.fprops = []
        self.bprops = []

        self.total = caffe.Timer()
        self.allrd = caffe.Timer()

        self.timedata={}

        os.chdir("C:\\Users\\seokh\\PycharmProjects\\PLATFORM6573_v1")
        self.solver = caffe.get_solver('model/cifar10_example/solver.prototxt')

        for nlayer in range(len(self.solver.net.layers)):
            self.fprops.append(caffe.Timer())
            self.bprops.append(caffe.Timer())


        self.solver.net.before_forward(lambda layer: self.fprops[layer].start())
        self.solver.net.after_forward(lambda layer: self.fprops[layer].stop())
        self.solver.net.before_backward(lambda layer: self.bprops[layer].start())
        self.solver.net.after_backward(lambda layer: self.bprops[layer].stop())

        self.solver.add_callback(lambda: self.total.start(), lambda: (self.total.stop(), self.allrd.start()))
        self.solver.add_callback(lambda: "", lambda: (self.allrd.stop(), self.show_time()))
        self.solver.add_callback(self.load, self.loss)
        self.solver.add_callback(self.load2, self.loss2)
        self.solver.add_callback(lambda :"", self.validation)



    def loss(self):
        if self.solver.iter % 20 == 0:
            print('[LOSS] ', self.solver.iter, self.solver.net.blobs['loss'].data)

    def load(self):
        if self.solver.iter %30 == 0:
            print('[LOAD] ',self.solver.iter, self.solver.net.blobs['loss'].data)

    def loss2(self):
        if self.solver.iter % 20 == 0:
            print('[LOSS2] ', self.solver.iter, self.solver.net.blobs['loss'].data)

    def load2(self):
        if self.solver.iter % 30 == 0:
            print('[LOAD2] ', self.solver.iter, self.solver.net.blobs['loss'].data)

    def stepIter(self, iter):
        for i in range(iter):
            start_time = time.time()
            self.solver.step(10)
            print('[%03d] Elapsed Time :: %f'%(i,time.time() - start_time))

    def validation(self):
        if self.solver.iter % 40 == 0:
            print('Emiting Test network')
            self.validation_data.emit(self.solver.test_nets[0])
            #self.vsolver.setValnet(self.solver.test_nets[0])
            #print('[VALIDATION]', end= ' ')
            #test_acc = []
            #for test_it in range(100):
            #    test_acc.append(self.solver.test_nets[0].forward()['accuracy'])
            #print(test_acc)
            #pool = Pool(processes=4)
            #test_netlist = [self.solver.test_nets[0] for i in range(10)]
            #pool.map(validatation_function, test_netlist)
            #pool.close()
            #pool.join()


    def show_time(self):
        if self.solver.iter % (self.solver.param.display) == 0:
            for i in range(len(self.solver.net.layers)):
                self.timedata[self.solver.net._layer_names[i]] = [self.fprops[i].ms, self.bprops[i].ms]
            print(self.timedata, self.total.ms, self.allrd.ms)


def validatation_function(test_net):
    test_acc = []
    for test_it in range(10):
        test_acc.append(test_net.forward()['accuracy'])
    return test_acc

class validationCore(QThread):

    def __init__(self, test_net = None, iter = 0):
        super().__init__()
        self.test_net = test_net
        self.iter = iter
        self.op_flag = False
        self.cnt = 0

    def __del__(self):
        self.wait()

    def run(self):
        test_acc = []
        while True:
            if self.op_flag:
                start_time = time.time()
                test_acc.append(self.test_net.forward()['accuracy'])
                self.cnt +=1
                if self.cnt == self.iter:
                    print(self.cnt, test_acc)
                    self.op_flag = False
                print('Testing [%d] iter process :: Elapsed Time %.5f'%(self.cnt, time.time() - start_time))
            else:
                self.msleep(10)

    @pyqtSlot(object)
    def setValnet(self, test_net):
        print('Setting Test netowkr')
        self.test_net = test_net
        self.op_flag = True
        self.cnt = 0



swc = SolverWithCallback()

# Validation Thread
vsolver = validationCore(iter = 100)
swc.validation_data.connect(vsolver.setValnet)
vsolver.start()


Total_timer_s = time.time()
swc.stepIter(10)
print('Total Elapsed Time is %.06f'%(time.time()-Total_timer_s))
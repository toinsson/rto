
import os
import time

import config

import filters

import numpy as np
import random




class MotionExplorer:
    """
    """
    def __init__(self,
        ndim = 2, 
        window=30, 
        order=4, 
        sr=128, 
        filter_cutoff=32, 
        logname='./'):
    """
    Parameters
    ----------
    ndim : int
        number of dimension of the input vector
    window : int
        windowing size for Savitzky Golay
    order : int
        order of the Savitzky Golay interpolation
    sr : int
        sampling rate
    filter_cutoff : int
        cutoff frequency for the Butter filter
    """
        self.axis = []
        self.axes = ndim

        self.order = order



        self.acceleration_limit = config.acceleration_limit
        self.lock = 0
        self.inhibited = False
        self.sr = sr
        self.last_at = None

        self.keep_level = config.keep_level
        self.knn_model = Knn()

        self.log = open(os.path.join(logname, "rawdata.txt"), "w", buffering=800000)

        for axis in range(self.axes):
            self.axis.append(filters.AxisFilter(window,order,sr=sr,cutoff=filter_cutoff))

        self.originality= config.originality
        self.k = config.k

        # self.sample_queue = Queue.Queue()
        self.invalidate()
        self.last_output = np.zeros((self.axes*self.order,))

    def invalidate(self):
        self.drops = 40


    def knn(self):
        # don't do anything if the motion is out of range!
        if self.inhibited:
            return 0, 0

        outputs = self.knn_model.classify(self.last_output)

        #mean of top k
        realscore = None
        score = None
        if outputs!=None and len(outputs)>0:
            realscore = np.mean(outputs[0:self.k])

            #discard NaN and inf
            if not (realscore>0 and realscore<1e6):
                realscore = 0

            #nonlinear scoring -- give points only if > threshold away
            if realscore>self.originality:
                score = realscore
            else:
                score = 0

        added = 0
        # only keep "original" points
        if self.axis[0].full and (not realscore or realscore>self.keep_level) and random.random()<config.knn_probability:
            self.knn_model.add_vector(self.last_output)
            added = 1

        return score,added


    def get_vectors(self):
        return self.knn_model.vectors

    # def continue_from(self, knn_state_fname):
    #     self.knn_model.load(knn_state_fname)

    def test_sample(self, s):
        (ax,ay,az) = s.acc()
        at = s.data_timestamp(SHAKE_SENSOR_ACC)
        if at!=self.last_at:
            print at,
            self.last_at = at

    def new_sample(self, ms, ndata):
        output = np.zeros(self.axes*self.order,)

        # do the sgolay fit
        for i, data in enumerate(ndata):
            self.axis[i].new_sample(ms, data)

        ## CHECK
        # la = max(max(ax,ay,az), -min(ax,ay,az))
        # if la>self.acceleration_limit:
        #     self.inhibited = True
        #     self.lock = 127.0

        # update packet drops
        if self.last_at!=None and ms-self.last_at>1:
            self.drops += ms-self.last_at

        self.drops *= 0.9
        self.last_at = ms
        self.lock = self.lock*0.95

        if self.lock<10:
            self.inhibited = False

        c =0
        for i in range(self.axes):
            d = self.axis[i].get_samples()
            #check if there actually is a new packet
            if len(d)>0:
                #only use the latest sample
                diffs = d[0]
                #copy out the derivatives
                for di in diffs:
                    output[c] = di
                    c = c + 1
        self.last_output = output

        return 1

class Knn:
    def __init__(self, preloaded=None):
        self.data = None
        self.icov = None
        self.vectors = 0

        if preloaded:
            infile = open(preloaded, 'r')
            (data,) = cPickle.load(infile)
            infile.close()

    def recompute_covariance(self):
        self.mean = np.mean(self.data, axis=0)
        self.icov = np.linalg.pinv(np.cov((self.data-self.mean).transpose()))

    def add_vector(self, vector):

        if self.data != None:

            self.data = np.vstack((self.data, vector))
            self.vectors += 1
            if self.vectors % 25 == 0:
                self.recompute_covariance()

        else:
            self.data = np.array([vector])
            self.mean = np.zeros((len(vector),))
            # self.icov = np.loadtxt("default.cov")
            # self.icov = np.ones()

    # def load(self, name):
    #     self.data = np.loadtxt(os.path.join(name, "vectors.data"))
    #     self.vectors = len(self.data)
    #     self.recompute_covariance()
    #     print "Continuing with %d vectors" % self.vectors

    def save(self,name):
        np.savetxt(os.path.join(name,"vectors.data" ), self.data)
        np.savetxt(os.path.join(name,"vectors.cov"), np.array(self.icov))


    def classify(self, input):

        ## useless
        if self.data==None or self.icov==None:
            return None

        d = self.data
        n = len(d)
        if n<0:
            return None
        input = input

        repmatrix = np.tile(input, (n,1) )

        #compute Mahalanobis distance
        diff = (d-repmatrix)
        sums = []

        for row in diff:
            row = row[:,np.newaxis]
            sums.append(np.sqrt(np.dot(np.dot(row.transpose(),self.icov),row)[0,0]))
        sums = np.array(sums)

        #sort neighbours
        ordered = np.sort(sums)
        return ordered

    def classify_from(self, input, index):

        if self.data==None or self.icov==None:
            return None

        #compute Mahalanobis distance
        diff = (self.data[index]-input)
        row = diff[:,np.newaxis]
        sums = []
        sums.append(np.dot(np.dot(row.transpose(),self.icov),row)[0,0])
        sums = np.array(sums)
        return sums

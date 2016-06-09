
import os
import time

import config

import filters

import numpy as np
import random



class MotionExplorer2:

    def __init__(self, 
        inputdim = 2, 
        
        window=30, 
        order=4, 
        sr=128, 
        filter_cutoff=32, 
        logname='./'):

        self.inputdim = inputdim
        self.order = order

        ## filtering
        self.axis = [filters.AxisFilter(window,order,sr=sr,cutoff=filter_cutoff) for _ in range(inputdim)]
        self.last_output = np.zeros(self.inputdim*self.order)

        ## observations space
        self.observations = np.zeros(self.inputdim*self.order)
        self.mean = np.zeros(self.inputdim*self.order)
        self.icov = np.eye(self.order*self.inputdim)


    def new_sample(self, ms, ndata):

        for i, data in enumerate(ndata):
            self.axis[i].new_sample(ms, data)

        ## CHECK
        # la = max(max(ax,ay,az), -min(ax,ay,az))
        # if la>self.acceleration_limit:
        #     self.inhibited = True
        #     self.lock = 127.0

        c =0
        for i in range(self.inputdim):
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

    def distance_to_observations(self, vector):
        """Return the Mahalanobis distance of vector to the space of all observations.
        The ouput distances are sorted.
        """
        diff = self.observations - vector
        distances = np.sqrt(np.diag(np.dot(np.dot(diff, self.icov), diff.T)))
        return np.sort(distances)

    def compute_observations_mean_icov(self):
        self.mean = np.mean(self.observations, axis=0)
        self.icov = np.linalg.pinv(np.cov((self.observations-self.mean).transpose()))




    # def add_vector(self, vector):

    #     if self.data != None:

    #         self.data = np.vstack((self.data, vector))
    #         self.vectors += 1
    #         if self.vectors % 5 == 0:
    #             self.recompute_covariance()
    #         # print 'knn1:', self.data

    #     else:
    #         self.data = np.array([vector])
    #         self.mean = np.zeros((len(vector),))
    #         # self.icov = np.loadtxt("default.cov")

    #         self.icov = np.eye(self.order*self.ndim)
    #         # print 'knn0: ' ,sorelf.data



class Knn:
    def __init__(self, ndim=2, order=4):
        self.data = None
        self.icov = None

        self.ndim = ndim
        self.order = order

        self.vectors = 0


        ## add vector 0
        self.add_vector(np.zeros(ndim*order))

    def recompute_covariance(self):
        self.mean = np.mean(self.data, axis=0)
        self.icov = np.linalg.pinv(np.cov((self.data-self.mean).transpose()))

    def add_vector(self, vector):

        if self.data != None:

            self.data = np.vstack((self.data, vector))
            self.vectors += 1
            if self.vectors % 5 == 0:
                self.recompute_covariance()
            # print 'knn1:', self.data

        else:
            self.data = np.array([vector])
            self.mean = np.zeros((len(vector),))
            # self.icov = np.loadtxt("default.cov")

            self.icov = np.eye(self.order*self.ndim)
            # print 'knn0: ' ,sorelf.data


    def distance_to_observations(self, vector):
        """Return the Mahalanobis distance of vector to the space of all observations.
        The ouput distances are sorted.
        """
        diff = self.data - vector
        distances = np.sqrt(np.diag(np.dot(np.dot(diff, self.icov), diff.T)))
        return np.sort(distances)


    # def classify(self, data):

    #     repmatrix = np.tile(data, (self.data.shape[0],1) )

    #     #compute Mahalanobis distance
    #     diff = (self.data-repmatrix)
    #     sums = []

    #     for row in diff:
    #         row = row[:,np.newaxis]
    #         sums.append(np.sqrt(np.dot(np.dot(row.transpose(),self.icov),row)[0,0]))

    #     sums = np.array(sums)

    #     #sort neighbours
    #     # ordered = np.sort(sums)
    #     # return ordered

    #     return sums

class MotionExplorer:
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
    def __init__(self, ndim = 2, window=30, order=4, sr=128, filter_cutoff=32, logname='./'):

        self.inputdim = ndim

        self.order = order
        self.acceleration_limit = config.acceleration_limit

        self.lock = 0
        self.inhibited = False
        self.sr = sr
        self.last_at = None

        self.knn_model = Knn(ndim=ndim, order=order)


        self.keep_level = config.keep_level
        self.log = open(os.path.join(logname, "rawdata.txt"), "w", buffering=800000)

        ## create the filter per axis
        self.axis = []
        for axis in range(self.inputdim):
            self.axis.append(filters.AxisFilter(window,order,sr=sr,cutoff=filter_cutoff))


        self.originality= config.originality
        self.k = config.k

        # def invalidate(self):
        # self.invalidate()
        self.drops = 40
        self.last_output = np.zeros((self.inputdim*self.order,))



    def new_sample(self, ms, ndata):
        output = np.zeros(self.inputdim*self.order,)

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
        for i in range(self.inputdim):
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
                score = realscore

        added = 0
        # only keep "original" points
        if (self.axis[0].full and 
            # (not realscore or realscore>self.keep_level) and 
            (realscore>self.keep_level) and 
            random.random()<config.knn_probability):

            self.knn_model.add_vector(self.last_output)
            added = 1




        return score,added


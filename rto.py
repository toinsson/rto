import numpy as np

class MotionExplorer:
    """
    Aim at exploring motions, represented as sampled observations of a n-dimensional input vector.
    This stream of vectors describe a vector space in which the Mahalanobis distance is used to 
    assess the distance of new samples to previously seen samples. Everytime a new sample is 
    observed that is when that K nearest neighbour are in average further away than N standard deviation, the new sample is deamed original and saved to the attribute observations. 
    """
    def __init__(self, inputdim = 2, stepsize = 10, order = 4, window = 30,
        start_buffer = 10, periodic_recompute = 5, number_of_neighbour = 5, 
        number_of_stdev = 4.5
        ):
        """
        Parameters
        ----------
        inputdim : int
            the number of dimension of the input vector.
        stepsize : int
            The size of the interpolation step in milliseconds.
        order : int
            The dimension of the output vector, 1 is position only, 2 includes velocity, 3 provides acceleration, and so on.
        window : int
            The size of the averaging window in samples.
        start_buffer : int
            The number of sample is takes before any observation can be saved, this leaves time 
            for the Savitsky Golay interpolation to start ouputing some data.
        periodic_recompute : int
            The number of samples after which mean and covarianve of saved observations will be recomputed.
        number_of_neighbour : int
            The number of closest neighnbours that are considered when assessing if a new sample is original or not.
        number_of_stdev : float
            The number of standard deviation a new vectors has to be from the mean of K nearest neighbour as measured by Mahalanobis distance. When the mean of K is greater than this value, the new sample is considered original and saved to observations.
        """
        self.inputdim = inputdim
        self.order = order

        ## filtering
        self.axis = [AxisFilter(stepsize, order, window) for _ in range(inputdim)]

        ## observations space
        self.observations = np.zeros((1,self.inputdim*self.order))
        self.mean = np.zeros(self.inputdim*self.order)
        self.icov = np.eye(self.inputdim*self.order)

        ## variable logic
        self.counter = 0
        self.start_buffer = start_buffer
        self.periodic_recompute = periodic_recompute

        self.number_of_neighbour = 5
        self.number_of_stdev = 4.5

        self.last_sample = np.zeros(self.inputdim*self.order)

    def new_sample(self, ms, ndata):
        """Passes a new observed sample to the motionexplorer. It will filter it based on the last 
        observed sample and compute the distance of this current sample to all previously saved 
        original samples. If the average distance of the N nearest neightbour is greater than X 
        stdev, then the current sample is saved to the class attribute observations. 

        Parameters
        ----------
        ms : int
            Timestamp in milliseconds. This can be easily produced with the time module and the 
            call to: int(round(time.time() * 1000)).
        ndata : iterable
            An iterable object (tuple, ndarray, ..) representing the N dimensional vector of the 
            current sample.

        Returns
        -------
        int, bool
            average Mahalanobis distance to the K nearest neighboour and flag saying if the 
            current sample is added to the set of original observations.
        """
        ## ndata.shape == inputdim
        self.counter += 1

        for i, data in enumerate(ndata):
            self.axis[i].new_sample(ms, data)

        ## recompute mean and icov every periodic_recompute
        if self.counter % self.periodic_recompute == 0:
            self.compute_observations_mean_icov()


        ## get last sample from each axis and squash to 1D
        sample = np.array([self.axis[i].samples[-1] for i in range(self.inputdim)]).reshape(-1)

        ## compute the distance of sample to all stored observations
        distances = self.distance_to_observations(sample)
        distance_meank = np.mean(distances[:self.number_of_neighbour])

        if (self.counter > self.start_buffer) and self.axis[0].full:

            ## keep the sample if further than number of stdev to previous observations
            if distance_meank > self.number_of_stdev:
                self.observations = np.vstack((self.observations, sample))
                added = True

            else: added = False

        else: 
            added = False

        self.last_sample = sample

        return distance_meank, added


    def distance_to_observations(self, vector):
        """Return the Mahalanobis distance of vector to the space of all observations.
        The ouput distances are sorted.
        """
        diff = self.observations - vector
        distances = np.sqrt(np.diag(np.dot(np.dot(diff, self.icov), diff.T)))
        return np.sort(distances)

    def compute_observations_mean_icov(self):
        self.mean = np.mean(self.observations, axis=0)
        # print self.observations.shape[0]
        if self.observations.shape[0] > 1:
            self.icov = np.linalg.pinv(np.cov((self.observations-self.mean).transpose()))


class AxisFilter:
    """Filters an unevenly sampled measurement dimension. It interpolates at constant time steps `stepsize` in ms, performs Butter worth filetering and Savitsky Golay interpolation of order `order` over a moving window `window`.
    """
    def __init__(self, stepsize, order, window):
        """
        Parameters
        ----------
        stepsize : int
            The size of the interpolation step in milliseconds.
        order : int
            The dimension of the output vector, 1 is position only, 2 includes velocity, 3 provides acceleration, and so on.
        window : int
            The size of the averaging window in samples.
        """
        self.stepsize = stepsize
        self.order = order

        self.interpolator = TimeInterpolator(stepsize)
        self.sgfitter = SavitskyGolayFitter(order, window)
        self.full = False

    def new_sample(self, time, value):
        self.samples = np.empty((0,self.order))

        self.interpolator.new_sample(time, value)

        for point in self.interpolator.value_steps:
            point = self.sgfitter.new_sample(point)
            self.samples = np.vstack((self.samples, point))

        self.full = self.sgfitter.full


class TimeInterpolator:
    """Interpolate between 2 measurements at constant step size X in ms.
    """
    def __init__(self, stepsize):
        self.stepsize = stepsize
        self.firstpoint = True

    def new_sample(self, time, value):

        if self.firstpoint == True:
            self.firstpoint = False
            self.time_steps = np.array([time])
            self.value_steps = np.array([value])

        else:
            self.time_steps = np.arange(self.last_time, time, self.stepsize)
            self.value_steps = np.interp(self.time_steps, [self.last_time, time], [self.last_value, value])

        self.last_time = time
        self.last_value = value


class SavitskyGolayFitter:
    def __init__(self, order = 4, window = 30):
        self.order = order

        if window%2==0:
            window = window + 1
        self.window = window

        #compute the savitzky-golay differentiators
        sgolay = self.savitzky_golay(order, window)
        self.sgolay_diff = []
        self.buffers = []
        self.samples = 0
        self.full = False

        #create the filters
        for i in range(order):
            self.sgolay_diff.append(np.ravel(sgolay[i, :]))
            self.buffers.append(IIRFilter(self.sgolay_diff[i], [1]))

    def new_sample(self, x):
        self.samples = self.samples + 1
        if self.samples>self.window:
            self.full = True
        fits = np.zeros((self.order,))

        # use enumerate or map
        c = 0
        for buffer in self.buffers:
            fits[c] = buffer.filter(x)
            c = c + 1

        return fits

    #sg coefficient computation
    def savitzky_golay(self, order = 2, window = 30):
        if window is None:
            window = order + 2

        if window % 2 != 1 or window < 1:
            raise TypeError("window size must be a positive odd number")
        if window < order + 2:
            raise TypeError("window size is too small for the polynomial")

        # A second order polynomial has 3 coefficients
        order_range = range(order+1)
        half_window = (window-1)//2
        B = np.mat(
            [ [k**i for i in order_range] for k in range(-half_window, half_window+1)] )

        M = np.linalg.pinv(B)
        return M


class IIRFilter:
    def __init__(self, B, A):
        """Create an IIR filter, given the B and A coefficient vectors.
        """
        self.B = B
        self.A = A
        if len(A)>2:
            self.prev_outputs = Ringbuffer(len(A)-1)
        else:
            self.prev_outputs = Ringbuffer(3)

        self.prev_inputs = Ringbuffer(len(B))

    def filter(self, x):
        """Take one sample and filter it. Return the output.
        """
        y = 0
        self.prev_inputs.new_sample(x)
        k =0
        for b in self.B:
            y = y + b * self.prev_inputs.reverse_index(k)
            k = k + 1

        k = 0
        for a in self.A[1:]:
            y = y - a * self.prev_outputs.reverse_index(k)
            k = k + 1

        y = y / self.A[0]

        self.prev_outputs.new_sample(y)
        return y

    def new_sample(self, x):
        return self.filter(x)


class Ringbuffer:
    def __init__(self, size, init=0):
        if size<1:
            throw(Exception("Invalid size for a ringbuffer: must be >=1"))
        self.n_samples = size
        self.samples = np.ones((size,))*init
        self.read_head = 1
        self.write_head = 0
        self.sum = 0

    def get_length(self):
        return self.n_samples

    def get_samples(self):
        return np.hstack((self.samples[self.read_head-1:],self.samples[0:self.read_head-1]))

    def get_sum(self):
        return self.sum

    def get_output(self):
        #self.read_head %= self.n_samples
        return self.samples[self.read_head-1]

    def get_mean(self):
        return self.sum / float(self.n_samples)

    def forward_index(self, i):
        new_index = self.read_head+i-1
        new_index = new_index % self.n_samples
        return self.samples[new_index]

    def reverse_index(self, i):
        new_index = self.write_head-i-1
        while new_index<0:
            new_index+=self.n_samples
        return self.samples[new_index]

    def new_sample(self, x):
        s = self.samples[self.write_head]
        self.samples[self.write_head] = x
        self.sum += x
        self.sum -= self.samples[self.read_head]
        self.read_head += 1
        self.write_head += 1
        self.read_head %= self.n_samples
        self.write_head %= self.n_samples
        return s

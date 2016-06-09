import numpy as np

class MotionExplorer:

    def __init__(self, inputdim = 2, stepsize = 10, order = 4, window = 30,
        start_buffer = 100, periodic_recompute = 5
        ):

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
        ## ndata.shape == inputdim
        self.counter += 1

        for i, data in enumerate(ndata):
            self.axis[i].new_sample(ms, data)

        ## recompute mean and icov every periodic_recompute
        if self.counter % self.periodic_recompute == 0:
            self.compute_observations_mean_icov()

        ## do nothing for start_buffer samples
        if self.counter < self.start_buffer:
            return 0, 0

        ## get last sample from each axis and squash to 1D
        sample = np.array([self.axis[i].samples[-1] for i in range(self.inputdim)]).reshape(-1)

        ## compute the distance of sample to all stored observations
        distances = self.distance_to_observations(sample)
        distance_meank = np.mean(distances[:self.number_of_neighbour])

        ## keep the sample if further than number of stdev to previous observations
        if distance_meank > self.number_of_stdev:
            self.observations = np.vstack((self.observations, sample))
            added = True

        else: added = False

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
        self.stepsize = stepsize
        self.order = order

        self.interpolator = TimeInterpolator(stepsize)
        self.sgfitter = SavitskyGolayFitter(order, window)

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

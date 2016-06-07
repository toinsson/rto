import numpy as np
import scipy
import scipy.signal


class AxisFilter:
    def __init__(self, window, order, sr=128, cutoff=32):
        self.order =  order
        self.interpolator = Interpolator()
        self.sgolay = SGolayFitter(window, order)
        self.pre_filter = ButterFilter(4, [cutoff/float(sr)])
        self.samples = []
        self.last_sample = []
        self.full = False

    def get_samples(self):
        return self.samples

    def new_sample(self, t, x):
        self.samples = []
        self.interpolator.add_packet(t, x)
        pkts = self.interpolator.get_packets()
        for new_packet in pkts:
            new_packet = self.pre_filter.new_sample(new_packet)
            self.samples.append(self.sgolay.new_sample(new_packet))
            self.full = self.sgolay.full
        return len(pkts)

    def new_sample_uninterpolated(self, x):
        new_packet = self.pre_filter.new_sample(x)
        self.last_sample = self.sgolay.new_sample(x)
        self.full = self.sgolay.full


class Interpolator:
    def __init__(self):

        self.first_time = True
        self.last_t = 0
        self.last_x = 0
        self.new_packets = []

    def add_packet(self, t, x):

        if self.first_time:
            self.first_time = False
            self.last_t = t
            self.last_x = x
            self.new_packets = [x]
            return

        dt = t - self.last_t

        if dt == 0:  # this packet isn't new
            self.new_packets = []
        elif dt==1:  # just one new packet
            self.new_packets = [x]
        else:  #linear interpolate
            self.new_packets = []
            for d in range(dt):
                a = float(d) / float(dt)
                i = (a)*self.last_x+(1-a)*x
                self.new_packets = [i] + self.new_packets

        self.last_t = t
        self.last_x = x

    def get_packets(self):
        return self.new_packets


class SGolayFitter:
    def __init__(self, window=30, order = 4):
        self.order = order

        if window%2==0:
            window = window + 1
        self.window = window
        #compute the savitzky-golay differentiators
        sgolay = self.savitzky_golay(window, order)
        self.sgolay_diff = []
        self.buffers = []
        self.samples = 0
        self.full = False
        #create the filters
        for i in range(order):
            self.sgolay_diff.append(np.ravel(sgolay[i, :]))
            self.buffers.append(Filter(self.sgolay_diff[i], [1]))

    def new_sample(self, x):
        self.samples = self.samples + 1
        if self.samples>self.window:
            self.full = True
        fits = np.zeros((self.order,))
        c = 0
        for buffer in self.buffers:
            fits[c] = buffer.filter(x)
            c = c + 1
        return fits

    #sg coefficient computation
    def savitzky_golay(self, window_size=None, order=2):
        if window_size is None:
            window_size = order + 2

        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window size is too small for the polynomial")

        # A second order polynomial has 3 coefficients
        order_range = range(order+1)
        half_window = (window_size-1)//2
        B = np.mat(
            [ [k**i for i in order_range] for k in range(-half_window, half_window+1)] )

        M = np.linalg.pinv(B)
        return M


class ButterFilter:
    def __init__(self, n, band, type='low'):
        ba = scipy.signal.butter(n, band, btype=type)
        self.filter = Filter(ba[0], ba[1])

    def new_sample(self, x):
        return self.filter.new_sample(x)


class Filter:
    def __init__(self, B, A):
        """Create an IIR filter, given the B and A coefficient vectors"""
        self.B = B
        self.A = A
        if len(A)>2:
            self.prev_outputs = Ringbuffer(len(A)-1)
        else:
            self.prev_outputs = Ringbuffer(3)

        self.prev_inputs = Ringbuffer(len(B))

    def filter(self, x):
        #take one sample, and filter it. Return the output
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


    def new_sample(self,x):
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


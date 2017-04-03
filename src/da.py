import os
import sys
import time
import os.path as pt
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import hlp
import pdb

class DA(object):
    """
    Denoising Auto-Encoder class (DA)
    """
    def __init__(
        self,
        np_rnd = None,
        n_vis =784,
        n_hid =500,
        th_rnd = None,
        t_w = None,
        t_bhid = None,
        t_bvis = None,
        tag = None
    ):
        """
        Initialize the DA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the DA and an MLP layer. When dealing with SdAs this always happens,
        the DA on layer 2 gets as input the output of the DA on layer 1,
        and the weights of the DA are used in the second stage of training
        to construct an MLP.

        """
        FT = theano.config.floatX

        if np_rnd is None:
            np_rnd = np.random.RandomState(120)

        # create a Theano random generator that gives symbolic random values
        if not th_rnd:
            th_rnd = RandomStreams(np_rnd.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        """
        # W is initialized with `initial_W` which is uniformely sampled
        # from -4*sqrt(6./(n_vis+n_hid)) and
        # 4*sqrt(6./(n_hid+n_vis))the output of uniform if
        # converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        """
        if not t_w:
            initial_W = np.asarray(
                np_rnd.uniform(
                    low=-4 * np.sqrt(6. / (n_hid + n_vis)),
                    high=4 * np.sqrt(6. / (n_hid + n_vis)),
                    size=(n_vis, n_hid)),
                dtype=theano.config.floatX)
            t_w = theano.shared(value=initial_W, name='W', borrow=True)

        if not t_bvis:
            t_bvis = theano.shared(value = np.zeros(n_vis, dtype = FT),
                name = 'b\'', borrow = True)

        if not t_bhid:
            t_bhid = theano.shared(value = np.zeros(n_hid, dtype = FT),
                name='b', borrow=True)

        self.t_w = t_w
        # b corresponds to the bias of the hidden
        self.t_b = t_bhid
        # b_prime corresponds to the bias of the visible
        self.t_b_prime = t_bvis
        # tied weights, therefore W_prime is W transpose
        self.t_w_prime = self.t_w.T

        self.th_rnd = th_rnd
        self.n_vis = n_vis
        self.n_hid = n_hid
        
        self.parm = [self.t_w, self.t_b, self.t_b_prime]

        if tag is None:
            self.tag = "{}-{}.da".format(n_vis, n_hid)
        else:
            self.tag = tag

    def __repr__(self):
        return self.tag

    def t_corrupt(self, t_x, t_lv):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the t_x
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.th_rnd.binomial(
            size = t_x.shape, n = 1,
            p = 1 - t_lv,
            dtype = theano.config.floatX) * t_x

    def t_encode(self, t_x):
        """
        Computes the values of the hidden layer
        """
        return T.nnet.sigmoid(T.dot(t_x, self.t_w) + self.t_b)

    def t_decode(self, t_x):
        """
        Computes the reconstructed input given the values of the
        hidden layer
        """
        return T.nnet.sigmoid(T.dot(t_x, self.t_w_prime) + self.t_b_prime)

    def f_train(self, t_x, t_corrupt = 0.2, t_rate = 0.1):
        """ return training function of the following signiture:
        input:
            lower and upper indices on training data
            alternative training data
        return:
            likelihood based cost
            square distance between training data and prediction
        
        """
        x = T.matrix('x')     # pipe data through this symble
        q = self.t_corrupt(x, t_corrupt)
        h = self.t_encode(q)
        z = self.t_decode(h)

        L = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)
        cost = T.mean(L)    # to be returned

        dist = T.mean(T.sqrt(T.sum((x - z) ** 2, axis = 1)))    # to be returned

        grad = T.grad(cost, self.parm)

        diff = [(p, p - t_rate * g) for p, g in zip(self.parm, grad)]

        t_fr = T.iscalar()
        t_to = T.iscalar()
        return theano.function(
            [t_fr, t_to],
            [cost, dist],
            updates = diff,
            givens = {x : t_x[t_fr:t_to]},
            name = "DA_trainer")
        
    def f_pred(self):
        T_x = T.matrix('x')
        T_h = self.t_encode(T_x)
        T_z = self.t_decode(T_h)
        F_pred = theano.function(
            [T_x],
            T_z)
        return F_pred

def test_da(da = None, x = None, tr = 0.1):

    if x is None:
        x = np.load(pt.expandvars('$AZ_TEST_IMG/lh001F1.npz'))['vtx']['tck']
        x = x.reshape(x.shape[0], -1)
        x = (x - x.min()) / (x.max() - x.min())
        x = theano.shared(x, borrow = True)
    
    # compute number of minibatches for training, validation and testing
    s_batch = 10
    n_batch = S_x.get_value(borrow=True).shape[0] / s_batch

    if da is None:
        np_rng = np.random.RandomState(120)
        da = DA(np_rnd = np_rng, n_vis = x.shape[1], n_hid = x.shape[1] * 4)

    ## -------- TRAINING --------
    train = da.f_train(t_x = S_x, t_corrupt = 0.2, t_rate = 0.05)
    start_time = time.clock()
    # go through training epochs
    for epoch in xrange(15):
        # go through trainng set
        c, d = [], []                     # cost, dist
        for i_batch in xrange(n_batch):
            r = train(i_batch * s_batch, (i_batch + 1) * s_batch)
            c.append(r[0])
            d.append(r[1])
        print 'Training epoch %d, cost %f, dist %f' % (epoch, np.mean(c), np.mean(d))

    end_time = time.clock()
    training_time = (end_time - start_time)
    print >> sys.stderr, ('ran for %.2fm' % (training_time / 60.))

    return da

if __name__ == '__main__':
    theano.config.floatX = 'float32'
    pass

from .da import DA
from . import hlp

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

theano.config.floatX = 'float32'


def __get_name__(t_x):
    if hasattr(t_x, 'name'):
        name = getattr(t_x, 'name')
    else:
        name = "x:{:d}".format(t_x.size / t_x.shape[0])
    if name is None:
        name = ""
    return name


def __set_name__(t_x, name):
    if hasattr(t_x, 'name'):
        setattr(t_x, 'name', name)


class SDA(list):
    """ Stacked denoising auto-encoder class (SdA) """

    def __init__(self, n_vis, n_hid=None, np_rnd=None, th_rnd=None):
        """ This class is made to support a variable number of layers. """
        np_rnd = np.random.RandomState(np_rnd)

        if not th_rnd:
            th_rnd = RandomStreams(np_rnd.randint(2**30))

        self.n_vis = n_vis  # number of visible feature
        self.np_rnd = np_rnd
        self.th_rnd = th_rnd

        if not n_hid:
            n_hid = ()
        if hasattr(n_hid, '__len__'):
            self.extend(n_hid)
        else:
            self.append(n_hid)

    def __mk_da__(self, n_hid):
        idx = len(self)
        if idx:
            n_vis = self[-1].n_hid
        else:
            n_vis = self.n_vis

        _da = DA(np_rnd=self.np_rnd,
                 th_rnd=self.th_rnd,
                 n_vis=n_vis,
                 n_hid=n_hid)

        _da.tag = "{0:d}:{1:d}-{2:d}".format(idx, n_vis, n_hid)
        _da.idx = len(self)
        return _da

    # override list.extend
    def extend(self, n_hds):
        return super(SDA, self).extend(
            self.__mk_da__(n_hid) for n_hid in n_hds)

    # override list.append
    def append(self, n_hid):
        return super(SDA, self).append(self.__mk_da__(n_hid))

    def __get_stack__(self, ly=None, ec=None, dc=None):
        if ly is None:
            ly = 0

        if ly < 0:
            ly = len(self) + ly

        if ec is None:
            lh = len(self)
        else:
            lh = min(ly + ec, len(self))

        if dc is None:
            lz = 0
        else:
            lz = max(lh - dc, 0)

        ec = self[ly:lh]
        dc = self[lz:lh]
        dc.reverse()
        return ec, dc

    def __get_parms__(self, ly=0, ec=None, dc=None):
        ec, dc = self.__get_stack__(ly, ec, dc)
        parms = []
        for a in ec:
            parms.append(a.t_w)
            parms.append(a.t_b)
        for a in dc:
            parms.append(a.t_b_prime)
        return parms

    def t_pipe(self, t_x, ly=0, ec=None, dc=None):
        ec, dc = self.__get_stack__(ly, ec, dc)
        name = __get_name__(t_x)

        # build pipe expression
        for a in ec:
            t_x = a.t_encode(t_x)
            name += "|{0:d}:{1:d}-{2:d}".format(a.idx, a.n_vis, a.n_hid)
        for a in dc:
            t_x = a.t_decode(t_x)
            name += "|{0:d}:{2:d}-{1:d}".format(a.idx, a.n_vis, a.n_hid)

        __set_name__(t_x, name)
        return t_x

    def t_encode(self, t_x, ly=0, dp=None):
        return self.t_pipe(t_x, ly, ec=dp, dc=0)

    def t_decode(self, t_x, ly=None, dp=None):
        if ly is None:
            ly = len(self)
        return self.t_pipe(t_x, ly, ec=0, dc=dp)

    def t_corrupt(self, t_x, lvl):
        return self.th_rnd.binomial(
            size=t_x.shape, n=1, p=1 - lvl, dtype=T.config.floatX) * t_x

    def f_encode(self, ly=0, dp=None):
        x = T.matrix('x')
        y = self.t_encode(x, ly, dp)
        return theano.function([x], y, name="SDA_encode")

    def f_train(self,
                x,
                y=None,
                corrupt=0.2,
                rate=0.1,
                lyr=None,
                ec=None,
                dc=None):

        t_x, t_y = hlp.to_shared(x, y)

        # request unsupervised training
        t_y = t_x if t_y is None else t_y

        x = T.matrix('x')  # a batch from t_x
        y = T.matrix('y')  # a batch from t_y

        # corrupted input
        q = self.t_corrupt(x, corrupt)

        # output at certian layer
        z = self.t_pipe(q, lyr, ec, dc)

        # cross entrophy
        cost = hlp.cross_entrophy(y, z, axis=1)

        # squared L2 norm
        dist = hlp.square_l2_norm(y, z, axis=1)

        parm = self.__get_parms__(lyr, ec, dc)

        grad = T.grad(cost, parm)

        diff = [(p, p - rate * g) for p, g in zip(parm, grad)]

        t_fr = T.iscalar()
        t_to = T.iscalar()
        return theano.function(
            [t_fr, t_to], [cost, dist],
            updates=diff,
            givens={x: t_x[t_fr:t_to],
                    y: t_y[t_fr:t_to]},
            name="SDA_trainer")

    def f_pred(self, ly=None, ec=None, dc=None):
        x = T.matrix('x')
        z = self.t_pipe(x, ly, ec, dc)
        return theano.function([x], z, name="SDA_pred")

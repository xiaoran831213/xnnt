import numpy as np
import theano
from copy import deepcopy as cp
from xutl.xdct import xdct
from xutl import lpz
from numpy.random import binomial, normal
from theano import function as F, tensor as T
from xnnt import exb, hlp
from xnnt.pcp import Pcp
from xnnt.cat import Cat
from xnnt.tnr.basc import Base as Tnr


FX = theano.config.floatX


class SW(Pcp):
    """ x --> sigmoid --> weighted sum. """
    def __init__(self, *par, **kwd):
        super(SW, self).__init__(*par, **kwd)

    def __expr__(self, x):
        _ = T.nnet.sigmoid(x)
        _ = T.dot(_, self.w) + self.b
        return _


class WS(Pcp):
    """ x --> weighted sum --> sigmoid. """
    def __init__(self, *par, **kwd):
        super(WS, self).__init__(*par, **kwd)

    def __expr__(self, x):
        _ = T.dot(x, self.w) + self.b
        _ = T.nnet.sigmoid(_)
        return _


def main(nep=20, **kw):
    """ testing ground. """
    tnx = kw.get('tnx', 0.00)
    lrt = kw.get('lrt', 1e-3)

    N, P, O, Q = 500, 100, 50, 10
    np.random.seed(1)
    # core effect
    x = normal(size=[N, P]).astype('<f4')
    ffq = 0.5                   # function frequency
    w = normal(size=[P, Q]) * binomial(1, ffq, [P, 1])
    xw = np.dot(x, w)
    s1 = np.std(xw, 0)

    rsq = 0.2                   # some noise
    s0 = s1 * np.sqrt(1 / rsq - 1)
    es = normal(np.zeros_like(xw), s0)
    y = xw + es

    # create the network
    nw0 = Pcp([P, O], s=1)
    nw0.s = 1
    nw0.w.set_value(normal(size=nw0.dim).astype('<f4'))

    nw1 = SW([O, O])
    nw1.w.set_value(normal(size=nw1.dim).astype('<f4'))

    nw2 = SW([O, Q])
    nw2.w.set_value(normal(size=nw2.dim).astype('<f4'))

    tn2 = None
    tn1 = None
    tn0 = None

    nwk = Cat([nw0, nw1, nw2])
    nwl = cp(nwk)

    for i in range(nep//5):
        x1 = nw0(x).eval()
        x2 = nw1(x1).eval()

        if tn2 is None:
            tn2 = Tnr(nw2, x2, y, err='L2', lr=lrt*3.0, bsz=N, tnx=tnx)
        else:
            tn2.hlt = 0
            tn2.xt.set_value(x2)
        print('terr 2:', tn2.terr())
        tn2.tune(15)
        print()

        y1 = tn2.xt.get_value()
        if tn1 is None:
            tn1 = Tnr(nw1, x1, y1, err='L2', lr=lrt*3.0, bsz=N, tnx=tnx)
        else:
            tn1.hlt = 0
            tn1.yt.set_value(y1)
            tn1.xt.set_value(x1)
        print('terr 1:', tn1.terr())
        tn1.tune(10)
        print()

        y0 = tn1.xt.get_value()
        if tn0 is None:
            tn0 = Tnr(nw0, x, y0, err='L2', lr=lrt*3.0, bsz=N)
        else:
            tn0.hlt = 0
            tn0.yt.set_value(y0)
        print('terr 0:', tn0.terr())
        tn0.tune(10)
        print()

    tnr = Tnr(nwl, x, y, err='L2', lr=lrt, bsz=N)
    tnr.tune(nep)
    return x, y, nwk, nwl

# x, y, n1, n2 = main(100, tnx=1.0)
# exb.L2(n1(x), y).mean().eval(); exb.L2(n2(x), y).mean().eval()
# exb.mcr(n1(x), y).mean().eval(); exb.mcr(n2(x), y).mean().eval()

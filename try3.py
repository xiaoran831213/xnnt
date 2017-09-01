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


def main(nep=20, **kw):
    """
    testing ground.
    """
    xtn = kw.get('xtn', 0.00)
    lrt = kw.get('lrt', 1e-3)
    
    N, P, O, S, Q = 500, 100, 50, 50, 10

    # core genomic effect
    x = normal(size=[N, P]).astype('<f4')

    ffq = 0.5
    w = normal(size=[P, Q]) * binomial(1, ffq, [P, 1])
    xw = np.dot(x, w)
    s1 = np.std(xw, 0)

    # add some noise
    rsq = 0.2
    s0 = s1 * np.sqrt(1 / rsq - 1)
    es = normal(np.zeros_like(xw), s0)
    y = xw + es

    # create network
    nw0 = Pcp([P, O], s=1)
    nw0.w.set_value(normal(size=nw0.dim).astype('<f4'))
    nw0.s = 1

    nw1 = Pcp([O, S], s=1)
    nw1.w.set_value(normal(size=nw1.dim).astype('<f4'))
    nw1.s = 1

    nw2 = Pcp([S, Q], s=1)
    nw2.w.set_value(normal(size=nw2.dim).astype('<f4'))
    nw2.s = 1

    tn2 = None
    tn1 = None
    tn0 = None

    nwk = Cat([nw0, nw1, nw2])
    nwl = cp(nwk)

    for i in range(nep):
        x1 = nw0(x).eval()
        x2 = nw1(x1).eval()

        if tn2 is None:
            tn2 = Tnr(nw2, x2, y, err='L2', lr=lrt, bsz=N, xtn=xtn)
        else:
            tn2.xt.set_value(x2)
        tn2.tune(10)

        y1 = tn2.xt.get_value()
        if tn1 is None:
            tn1 = Tnr(nw1, x1, y1, err='L2', lr=lrt, bsz=N, xtn=xtn)
        else:
            tn1.yt.set_value(y1)
            tn1.xt.set_value(x1)
        tn1.tune(10)

        y0 = tn1.xt.get_value()
        if tn0 is None:
            tn0 = Tnr(nw0, x, y0, err='L2', lr=lrt, bsz=N)
        else:
            tn0.yt.set_value(y0)
        tn0.tune(10)
        
    tnr = Tnr(nwl, x, y, err='L2', lr=lrt, bsz=N)
    tnr.tune(nep)
    return x, y, nwk, nwl

# exb.L2(n1(x), y).mean().eval(); exb.L2(n2(x), y).mean().eval()

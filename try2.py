import numpy as np
import theano
from xutl.xdct import xdct
from xutl import lpz
from numpy.random import binomial, normal
from theano import function as F, tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from time import time as tm
from xnnt import exb, hlp
from xnnt.mlp import MLP
from xnnt.tnr.basc import Base as Tnr
from xnnt.hlp import C, S, parms
import pandas as pd
import sys

FX = theano.config.floatX


def main(fi='dat/0013', nep=0):
    """
    testing ground.
    """
    dat = xdct(lpz(fi))
    gmx = dat.gmx.astype('f')
    dsg = gmx.sum(1)
    x = gmx.reshape(dsg.shape[0], -1)
    N, P = x.shape[0], x.shape[1]
    Q = 100

    # core genomic effect
    ffq = .5
    b = normal(size=[dsg.shape[1], Q]) * binomial(1, ffq, [dsg.shape[1], 1])
    x_b = np.dot(dsg, b)
    sd1 = np.std(x_b, 0)

    # add some noise
    rsq = .3
    sd0 = sd1 * np.sqrt(1 / rsq - 1)
    eps = np.random.normal(np.zeros_like(x_b), sd0)
    y = x_b + eps

    # training, validation, and generalization sets.
    div = int(0.8 * N)
    xD, xE = x[:div], x[div:]
    yD, yE = y[:div], y[div:]

    div = int(0.8 * xD.shape[0])
    xT, xV = xD[:div], xD[div:]
    yT, yV = yD[:div], yD[div:]
    dim = [P, 500, 500, 500, Q]

    # gx = T.grad(err, x).eval()
    # x.set_value(x.get_value() - gx * 0.01)
    # err.eval()
    nwk = MLP(dim)
    nwk[-1].s = 1

    lrt = 1e-3

    xls = [hlp.S(xT)]
    for i, n in enumerate(nwk[:-1]):
        __x = hlp.S(n(xls[i]).eval())
        xls.append(__x)

    __y = yT
    for i in reversed(range(1, len(xls) + 1)):
        print(i)
        if i > 1:
            __x = nwk.sub(0, i - 1)(xT).eval()
        else:
            __x = xT
        __n = nwk.sub(i - 1, i)

        __e = 'L2' if __n[-1].s == 1 else 'CE'
        __t = Tnr(__n, __x, __y, err=__e, lr=lrt, bsz=xT.shape[0])

        __t.tune(5)
        __y = __t.xt.get_value()

    return x, y
    # tnr = Tnr(nwk, xT, yT, xV, yV, xg=xE, yg=yE, err='L2', lr=lrt)
    # tnr.tune(nep)
    # return x, y, tnr

import numpy as np
import theano
from xutl.xdct import xdct
from xutl import lpz
from theano import function as F, tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from time import time as tm
from xnnt import exb
from xnnt.mlp import MLP
from xnnt.tnr.basb import Base as Tnr
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

    ffq = .5
    x_b = np.dot(dsg, np.random.binomial(1, ffq, (dsg.shape[1], 1)))
    sd1 = np.std(x_b)

    rsq = .3
    sd0 = sd1 * np.sqrt(1/rsq - 1)
    y = x_b + np.random.normal(0, sd0, (N, 1))

    div = int(0.8 * N)
    xD, xE = x[:div], x[div:]
    yD, yE = y[:div], y[div:]

    div = int(0.8 * xD.shape[0])
    xT, xV = xD[:div], xD[div:]
    yT, yV = yD[:div], yD[div:]

    dim = [P, 1000, 500, 500, 1]

    nwk = MLP(dim)
    nwk[-1].s = 1

    tnr = Tnr(nwk, xT, yT, xV, yV, xg=xE, yg=yE, err='L2', lr=1e-4)
    tnr.tune(nep)
    return x, y, tnr

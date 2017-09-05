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
    xnp = kw.get('xnp', 10)
    lrt = kw.get('lrt', 1e-3)

    N, P, O, Q = 1000, 100, 50, 10
    np.random.seed(1)

    # core effect
    x = normal(size=[N, P]).astype('<f4')
    ffq = 0.2                   # function frequency
    w = normal(size=[P, Q]) * binomial(1, ffq, [P, 1])
    xw = np.dot(x, w)
    s1 = np.std(xw, 0)

    rsq = 0.1                   # some noise
    s0 = s1 * np.sqrt(1 / rsq - 1)
    es = normal(np.zeros_like(xw), s0)
    y = xw.astype('<f4') + es.astype('<f4')

    # create the network
    ns = [Pcp([P, O], s=1)]
    for i in range(2):
        ns.append(SW([O, O]))
    ns.append(SW([O, Q]))

    for n in ns:
        n.w.set_value(normal(size=n.dim).astype('<f4'))

    nw1 = Cat(ns)               # top down stepwise training
    nw2 = cp(nw1)               # traditional training
    ts = [None] * len(ns)
    for i in range(nep // xnp):
        # forward pass:
        xs = [x]
        for n in ns[:-1]:
            xs.append(n(xs[-1]).eval())

        yj = y
        for j in reversed(range(len(ns))):
            xj = xs[j]
            if ts[j] is None:
                ts[j] = Tnr(ns[j], xj, yj, err='L2', lr=lrt, bsz=N, tnx=tnx)
                if j == 0:
                    ts[j].tnx = 0
            else:
                ts[j].yt.set_value(yj)
                ts[j].xt.set_value(xj)
                print('terr', j, ts[j].terr())
            ts[j].tune(xnp)
            df = np.abs(ts[j].xt.get_value() - xj).sum()
            print('xdiff:', j, df)
            yj = ts[j].xt.get_value()

    # tnr = Tnr(nw2, x, y, err='L2', lr=lrt, bsz=N)
    # tnr.tune(nep)
    return x, y, nw1, nw2

# x, y, n1 = main(100, tnx=1.0)
# exb.L2(n1(x), y).mean().eval(); exb.L2(n2(x), y).mean().eval()
# exb.mcr(n1(x), y).mean().eval(); exb.mcr(n2(x), y).mean().eval()

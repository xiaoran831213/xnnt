import numpy as np
import theano
from theano import function as F, tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from time import time as tm
from xnnt import exb
from xnnt.hlp import C, S, parms
import pandas as pd
import sys
from xnnt.tnr import basb


def main():
    """ testing ground. """
    dat = np.load('../dat/0013.npz')
    return dat

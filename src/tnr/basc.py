import numpy as np
import theano
from theano import function as F, tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from time import time as tm
from xnnt import exb
from xnnt.hlp import C, S, parms
import re
import pandas as pd
import sys

FX = theano.config.floatX


class Base(object):
    """
    Class for neural network training.
    """

    def __init__(self, nwk, xt, yt=None, xv=None, yv=None, *arg, **kwd):
        """
        : -------- parameters -------- :
        nwk: an expression builder for the neural network to be trained,
        could be a Nnt object.

        xt: the inputs, with the first dimension standing for sample units.
        If unspecified, the trainer will try to evaluate the entry point and
        cache the result as source data.

        yt: the labels, with the first dimension standing for sample units.
        if unspecified, a simi-unsupervied training is assumed as the labels
        will be identical to the inputs.

        xv: the valication data inputs
        yv: the validation data labels

        : -------- kwd: keywords -------- :
        ** dataset for generalization evaluation:
        **   xg: the inputs
        **   yg: the labels
        **
        ** trainer configuration:
        **  bsz: batch size.
        **   lr: learning rate.
        **  mmt: momentom

        **  err: expression builder of training errors
        **  reg: expression builder of weight regulator.
        **  lmb: weight decay factor, the lambda

        **  vdr: validation disruption rate
        **  hte: the halting training error.
        """
        # numpy random number generator
        seed = kwd.pop('seed', None)
        nrng = kwd.pop('nrng', np.random.RandomState(seed))
        trng = kwd.pop('trng', RandomStreams(nrng.randint(0x7FFFFFFF)))

        # private members
        self.__seed__ = seed
        self.__nrng__ = nrng
        self.__trng__ = trng

        # expression of error and regulator terms
        err = getattr(exb, kwd.get('err', 'CE'))
        reg = getattr(exb, kwd.get('reg', 'L1'))

        # the validation disruption
        self.vdr = S(kwd.get('vdr'), 'VDR')

        # the denoising
        self.dns = S(kwd.get('dns'), 'DNS')

        # the halting rules, and halting status
        self.hte = kwd.get('hte', 1e-3)  # 1. low training error
        self.hgd = kwd.get('htg', 1e-7)  # 2. low gradient
        self.hlr = kwd.get('hlr', 1e-7)  # 3. low learning rate
        self.hvp = kwd.get('hvp', 100)  # 4. out of patient
        self.hlt = 0

        # current epoch index, use int64
        ep = kwd.get('ep', 0)
        self.ep = S(ep, 'EP')

        # training batch ppsize, use int64
        bsz = kwd.get('bsz', 20)
        self.bsz = S(bsz, 'BSZ')

        # current batch index, use int64
        self.bt = S(0, 'BT')

        # momentumn, make sure momentum is a sane value
        self.mmt = S(kwd.get('mmt', None), 'MMT')

        # learning rate
        lr = kwd.get('lr', 0.01)  # learning rate
        acc = kwd.get('acc', 1.02)  # acceleration
        dec = kwd.get('dec', 0.95)  # deceleration
        self.lr = S(lr, 'LRT', 'f')
        self.acc = S(acc, 'ACC', 'f')
        self.dec = S(dec, 'DEC', 'f')

        # weight decay, lambda
        lmd = kwd.get('lmd', None)
        self.lmd = None if lmd is None else S(lmd, 'LMD', 'f')

        # the neural network
        self.nwk = nwk

        # inputs and labels, for modeling, validation, and generalization
        self.xt = S(xt, FX)
        self.yt = S(yt, FX)
        self.xv = S(xv, FX)
        self.yv = S(yv, FX)
        self.xg = S(kwd.get('xg', None), FX)
        self.yg = S(kwd.get('yg', None), FX)

        # -------- construct trainer function -------- *
        # 1) symbolic expressions
        x = T.tensor(name='x', dtype=FX, broadcastable=self.xt.broadcastable)
        y = T.tensor(name='y', dtype=FX, broadcastable=self.yt.broadcastable)

        # prediction
        pred = nwk(x)  # generic

        # mean correlation between label {yv} and prediction of all features
        pcor = exb.mcr(nwk(x), y)

        # list of symbolic parameters to be tuned
        pars = parms(pred)

        # unlist symbolic weights into a vector
        vwgt = T.concatenate([p.flatten() for p in pars if p.name == 'w'])

        # symbolic batch cost, which is the mean trainning erro over all
        # observations and sub attributes.

        # The observations are indexed by the first dimension of y, the last
        # dimension indices data entries for each observation,
        # e.g. voxels in an MRI region, and SNPs in a gene.
        # The objective function, err, returns a scalar of training loss, it
        # can be the L1, L2 norm and CE.
        erro = err(pred, y).mean()

        # the sum of weights calculated for weight decay.
        wsum = reg(vwgt)
        cost = erro if self.lmd is None else erro + wsum * self.lmd

        # symbolic gradient of cost WRT parameters
        grad = T.grad(cost, pars)
        gvec = T.concatenate([g.flatten() for g in grad])
        gabs = T.abs_(gvec)
        gsup = T.max(gabs)

        xgrad = T.grad(cost, x)
        xgabs = T.abs_(xgrad)
        xgsup = T.max(xgabs)

        # trainer control
        nwep = ((self.bt + 1) * self.bsz) // self.xt.shape[-2]  # new epoch?

        # 2) define updates after each batch training
        up = []

        # update parameters using gradiant decent, and momentum
        for p, g in zip(pars, grad):
            # use momentum
            if self.mmt is not None:
                # initialize accumulated gradient
                h = S(np.zeros_like(p.get_value()))
                # update momentum
                up.append((h, self.mmt * h + (1 - self.mmt) * g))
                # step down the accumulated gradient
                up.append((p, p - self.lr * h))
            # normal settings
            else:
                up.append((p, p - self.lr * g))

        # update batch and eqoch index
        up.append((self.bt, (self.bt + 1) * (1 - nwep)))
        up.append((self.ep, self.ep + nwep))

        # 3) the trainer functions
        # expression of batch and whole data feed:
        _ = T.arange((self.bt + 0) * self.bsz, (self.bt + 1) * self.bsz)

        # batch data and whole data
        if self.dns:            # 1) denoise settings
            msk = self.__trng__.binomial(
                self.yt.shape, 1, 1 - self.dns, dtype=FX)
            bts = {x: self.xt.take(_, 0, 'wrap'),
                   y: self.yt.take(_, 0, 'wrap') * msk.take(_, -2, 'wrap')}
            dts = {x: self.xt, y: self.yt * msk}
        else:                   # .) normal settings
            bts = {x: self.xt.take(_, 0, 'wrap'),
                   y: self.yt.take(_, 0, 'wrap')}
            dts = {x: self.xt, y: self.yt}

        # stochastic gradient descent by batch
        self.step = F([], cost, name="step", givens=bts, updates=up)

        # training error, training cost
        self.terr = F([], erro, name="terr", givens=dts)
        self.tcor = F([], pcor, name="tcor", givens=dts)
        self.tauc = lambda: self.pauc(self.xt, self.yt)
        self.tcst = F([], cost, name="tcst", givens=dts)

        # weights, and parameters
        self.wsum = F([], wsum, name="wsum")
        self.gsup = F([], gsup, name="gsup", givens=dts)
        # * -------- done with trainer functions -------- *
        self.xgsp = F([], xgsup, name="xgsp", givens=dts)
        self.xgrd = F([], xgrad, name="xgrd", givens=dts)

        # * -----  validation and generalization  ------- *
        # validation dataset
        if (self.xv and self.yv) is not None:
            # enable validation disruption (for binomial only)?
            if self.vdr:
                yds = self.__trng__.binomial(
                    self.yv.shape, 1, self.vdr, dtype=FX)
                gvn = {x: self.xv, y: (self.yv + yds) % C(2.0, FX)}
            else:
                gvn = {x: self.xv, y: self.yv}

            # error, correlation, and AUC
            self.verr = F([], erro, name="verr", givens=gvn)
            self.vcor = F([], pcor, name="vcor", givens=gvn)
            self.vauc = lambda: self.pauc(self.xv, self.yv)

        # generalization dataset
        if (self.xg and self.yg) is not None:
            gvn = {x: self.xg, y: self.yg}
            self.gerr = F([], erro, name="gerr", givens=gvn)
            self.gcor = F([], pcor, name="gcor", givens=gvn)
            self.gauc = lambda: self.pauc(self.xg, self.yg)

        # * ---------- recording tracks and history  ---------- *
        self.__trac__ = {}

        hd, exc = {}, ['step', 'gvec', 'tune', 'xgrd']
        inc = ['tauc', 'vauc', 'gauc', 'vcor', 'gcor']
        for k, v in vars(self).items():
            if k.startswith('__') or k in exc:
                continue
            # possible theano shared gpu variable
            if isinstance(v, (type(self.lr), type(self.ep))) and v.ndim < 1:
                hd[k] = v.get_value
            elif isinstance(v, (type(self.step), float, int, str, bool)):
                hd[k] = v
            elif k in inc:
                hd[k] = v
            else:
                pass

        self.__head__ = list(hd.items())
        self.__time__ = .0

        # the initial record, and history
        self.__hist__ = [self.__rpt__()]
        self.__emin_verr__ = self.__hist__[-1]  # min verr
        self.__emin_terr__ = self.__hist__[-1]  # min terr

        # printing format, also consider suppression
        self.nptr = kwd.get('nptr', ['.*auc'])
        _np = [re.compile(_) for _ in self.nptr]
        self.__pfmt__ = "{ep:04d}: {tcst:.1e}"
        if not any(re.match(_, 'tcor') for _ in _np):
            self.__pfmt__ += '{tcor:8.1e}'
        if not any(re.match(_, 'tauc') for _ in _np):
            self.__pfmt__ += '{tauc:8.1e}'
        # 2) weight decay
        if self.lmd is not None:
            if not any(re.match(_, 'lmd') for _ in _np):
                self.__pfmt__ += '={terr:.1e} + {lmd:.1e}*{wsum:.1e}'
        # 4) validation
        if (self.xv and self.yv) is not None:
            self.__pfmt__ += '|{verr:.1e}'
            if not any(re.match(_, 'vcor') for _ in _np):
                self.__pfmt__ += '{vcor:8.1e}'
            if not any(re.match(_, 'vauc') for _ in _np):
                self.__pfmt__ += '{vauc:8.1e}'
        # .) generalization
        if (self.xg and self.yg) is not None:
            self.__pfmt__ += '|{gerr:.1e}'
            if not any(re.match(_, 'gcor') for _ in _np):
                self.__pfmt__ += '{gcor:8.1e}'
            if not any(re.match(_, 'gauc') for _ in _np):
                self.__pfmt__ += '{gauc:8.1e}'
        # .) sum of gradient, learning rate
        self.__pfmt__ += "@{gsup:.1e} {xgsp:.1e} {lr:.1e}"

    # -------- helper funtions -------- *
    def nbat(self):
        return self.xt.get_value().shape[-2] // self.bsz.get_value()

    def pred(self, input=None, evl=True):
        """
        Predicted outcome given {input}. By default, use the training samples.
        input: network input
        evl: evaludate the expression (default = True)
        """
        if input is None:
            input = self.xv
        hat = self.nwk(input)
        return hat.eval() if evl else hat

    def pauc(self, x=None, y=None):
        """ AUC on validation dataset.
        no sanity check is imposed"""
        from sklearn.metrics import roc_auc_score
        if x is None or y is None:
            x = self.xt
            y = self.yt

        if isinstance(x, (type(self.xt), type(self.ep))):
            x = x.get_value()
        if isinstance(x, T.TensorConstant):
            x = x.data
        if isinstance(x, T.TensorVariable):
            x = x.eval()

        if isinstance(y, (type(self.xt), type(self.ep))):
            y = y.get_value()
        if isinstance(y, T.TensorConstant):
            y = y.data
        if isinstance(y, T.TensorVariable):
            y = y.eval()
        y = y.ravel()
        
        p = self.pred(x).ravel()
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = 0.0
        return auc

    # * -------- extra getter and setters -------- *
    def time(self, val=None) -> "float, scalar":
        """ Time elapsed since the start of training. """
        if val is not None:
            self.__time__ = val
        return self.__time__

    def __rpt__(self):
        """ report current status. """
        r = dict()
        for k, v in self.__head__:
            if callable(v):
                v = v()
                if isinstance(v, np.ndarray):
                    v = v.item()
            r[k] = v

        r['time'] = self.__time__
        return r

    def __onep__(self):
        """ called on new epoch. """
        # update semi snap shots
        r = self.__hist__[-1]
        if self.xv is not None and self.yv is not None:
            if r['verr'] < self.__emin_verr__['verr']:  # min verr
                self.__emin_verr__ = r
            if r['terr'] < self.__emin_terr__['terr']:  # min terr
                self.__emin_terr__ = r

    def __onbt__(self):
        """ called on new batch """
        pass

    def __stop__(self):
        """ return true to signal training stop. """
        # already halted?
        if self.hlt:
            return True

        # no training histoty, do not stop.
        if len(self.__hist__) < 1:
            return False

        # pull out the latest history
        r = self.__hist__[-1]

        # terr flucturation, do not stop!
        if r['terr'] > self.__emin_terr__['terr']:
            return False

        # check the latest history
        if r['terr'] < self.hte:  # early stop on terr
            self.hlt = 1
            return True
        if r['gsup'] < self.hgd:  # converged
            self.hlt = 2
            return True
        # early stop if we run out of patient on rising verr
        if r['ep'] - self.__emin_verr__['ep'] > self.hvp:
            self.hlt = 3
            return True
        if r['lr'] < self.hlr:  # fail
            self.hlt = 9
            return True
        return False

    def tune(self, nep=1, nbt=0, rec=0, prt=0):
        """ tune the parameters by running the trainer {nep} epoch.
        nep: number of epoches to go through
        nbt: number of extra batches to go through

        rec: frequency of recording. 0 means record after each epoch,
        which is the default, otherwise, record for each batch, which
        can be time consuming.

        prt: frequency of printing.
        """
        bt = self.bt
        b0 = bt.eval().item()  # starting batch
        ei, bi = 0, 0  # counted epochs and batches

        nep = nep + nbt // self.nbat()
        nbt = nbt % self.nbat()

        while (ei < nep or bi < nbt) and not self.__stop__():
            # send one batch for training
            t0 = tm()
            # update x
            self.xt.set_value(
                 self.xt.get_value() - self.xgrd())

            self.step()
            # self.__onbt__()  # on each batch
            self.__time__ += tm() - t0

            # update history
            bi = bi + 1  # batch count increase by 1
            if rec > 0:  # record each batch
                self.__hist__.append(self.__rpt__())
            if prt > 0:  # print each batch
                print(self)

            # see the next epoch relative to batch b0?
            if bt.get_value().item() == b0:
                ei = ei + 1  # epoch count increase by 1
                bi = 0  # reset batch count

            # after an epoch
            if bt.get_value().item() == 0:
                # record history
                if rec == 0:  # record at new epoch
                    self.__hist__.append(self.__rpt__())
                # print
                if prt == 0:  # print
                    print(self)

                self.__onep__()  # on each epoch
            sys.stdout.flush()

    def __str__(self):
        """ trainer status in short string. """
        return self.__pfmt__.format(**self.__hist__[-1])

    def hist(self):
        """ the training history in a pandas data frame. """
        return pd.DataFrame(self.__hist__)

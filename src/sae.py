from xnnt.ae import AE
from xnnt.cat import Cat
from xnnt.mlp import MLP


class SAE(Cat):
    """
    Stacked Auto Encoder
    """
    def __init__(self, fr, **kwd):
        """
        Initialize the stacked auto encoder by a list of code dimensions.
        The weight and bias terms are initialized by default rule.

        -- fr: the series input to build SAE from. can be a series of int
        or AEs.
        """
        # by a list of integers
        if all(isinstance(_, int) for _ in fr):
            fr = list(fr)
            sa = [AE(_, **kwd) for _ in zip(fr[:-1], fr[1:])]
        # by a list of AEs
        elif all(isinstance(_, AE) for _ in fr):
            fr = list(fr)
            sa = fr
        # by a MLP
        elif isinstance(fr, MLP):
            sa = [AE(_, **kwd) for _ in fr]
        else:
            raise ValueError('SAE: fr must be int[2+], AE[2+] or MLP')
        ec = [a.ec for a in sa]
        dc = [a.dc for a in reversed(sa)]
        super(SAE, self).__init__(ec + dc)

        self.sa = sa            # default view
        self.ec = MLP(ec)       # encoder view
        self.dc = MLP(dc)       # decoder view

    def sub(self, start=None, stop=None, copy=False):
        """ get sub stack from of lower encoding stop
        -------- parameters --------
        start: starting level of sub-stack extraction. by default
        always extract from the least encoded level.

        stop: stopping level of the sub-stack, which will be cropped
        to the full depth of the original stack if exceeds it.

        copy: should the sub stack deep copy the parameters?
        """
        ret = SAE(self.sa[start:stop])
        if copy:
            import copy
            ret = copy.deepcopy(ret)
        return ret

    @staticmethod
    def Train(w, x, u=None, nep=20, gdy=0, **kwd):
        """ train the stacked autoencoder {w}.
        w: the stacked autoencoders
        x: the training features.
        u: the training labels.

        nep: number of epochs to go through, if not converge.
        gdy: number of greedy pre-training to go through per added layer.

        kwd - additional key words to pass on to the trainer.
        ** lrt: initial learning rate.
        ** hte: halting error
        **   v: the testing features.
        **   y: the testing label.

        returns: the used trainer {class: xnnt.tnr.bas}.
        """
        # the trainer class
        from xnnt.tnr.cmb import Comb as Tnr

        # layer-wise greedy pre-training (incremental).
        if gdy > 0:
            for i in range(1, len(w.sa)):
                sw = w.sub(0, i)
                print('pre-train sub-stack:', sw)
                tr = Tnr(sw, x, u=u)
                tr.tune(gdy)

        # whole stack fine-tuning
        print('train stack:', w)
        tr = Tnr(w, x, u=u, **kwd)
        tr.tune(nep)

        return tr

from xnnt.ae import AE
from xnnt.cat import Cat


class SAE(Cat):
    """
    Stacked Auto Encoder
    """
    def __init__(self, AEs=None, **kwd):
        """
        Initialize the stacked auto encoder by a list of code dimensions.
        The weight and bias terms in the AEs are initialized by default rule.
        -------- optional parameters --------
        -- AEs: a list of autoencoders.
        ** dim: a list of dimensions to create autoencoders.
        """
        super(SAE, self).__init__()
        self.sa = []
        self.ec = Cat()
        self.dc = Cat()
        self.stack(AEs, **kwd)

    def stack(self, AEs=[], **kwd):
        """ stack a list of AEs on top of the current SAE.
        AEs: autoencoders to be stacked, can be an SAE form, or a list of
        dimensions to build an SAE, otherwise no action will be taken.
        """
        # list of AE from an SAE, or dimension numbers.
        if AEs and all([isinstance(a, int) for a in AEs]):
            sa = [AE(d, **kwd) for d in zip(AEs[:-1], AEs[1:])]
        elif isinstance(AEs, SAE):
            sa = AEs.sa
        else:
            sa = AEs

        # list of encoders and decoders to be stacked
        ec = [a.ec for a in sa]
        dc = [a.dc for a in reversed(sa)]

        # stack them now.
        self.sa.extend(sa)
        self.ec.extend(ec)
        self.dc[:0] = dc
        self[:] = self.ec + self.dc

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

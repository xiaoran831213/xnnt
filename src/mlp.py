from xnnt.pcp import Pcp
from xnnt.cat import Cat


class MLP(Cat):
    """
    Multiple Layered Perceptron
    """
    def __init__(self, fr, **kw):
        """
        Initialize the stacked auto encoder by a list of code dimensions.
        The weight and bias terms in the AEs are initialized by default rule.
        -------- parameters --------
        fr: from what to build the MLP, can be a series of Pcp or integers.
        """
        if all(isinstance(_, int) for _ in fr):
            nts = [Pcp(_, **kw) for _ in zip(fr[:-1], fr[1:])]
        elif all(isinstance(_, Pcp) for _ in fr):
            nts = fr
        else:
            raise ValueError("MLP: fr must be int[2+] or Pcp[1+]")
        super(MLP, self).__init__(nts)

    def sub(self, start=None, stop=None, copy=False):
        """ get sub stack from of lower encoding stop
        -------- parameters --------
        start: starting layer of sub MLP, the default is the lowest layer.

        stop: stopping layer of the sub MLP, will be cropped to the full
        depth of the original MLP.

        copy: the sub MLP is to be deeply copied, instead of shairing
        components of the current MLP.
        """
        ret = MLP(self[start:stop])
        if copy:
            import copy
            ret = copy.deepcopy(ret)
        return ret

    @staticmethod
    def Train(nwk, x, y, u=None, v=None, nep=20, gdy=0, **kw):
        """ train the MLP
        nwk: the MLP
        x: the training features.
        y: the training lables.

        nep: number of epochs to go through, if not converge.
        gdy: number of greedy pre-training to go through per layer.

        kw - additional key words to pass on to the trainer.
        ** lrt: initial learning rate.
        ** hte: halting error
        **   u: the testing features.
        **   v: the testing label.

        returns: the used trainer {class: xnnt.tnr.bas}.
        """
        # the trainer class
        from xnnt.tnr.cmb import Comb as Tnr

        # layer-wise greedy pre-training (incremental).
        if gdy > 0:
            print('MLP: pre-train:', nwk)
            from xnnt.sae import SAE
            pkw = kw.copy()
            pkw['err'] = 'CE'
            pkw.pop('u', None)
            pkw.pop('v', None)
            xi = x
            i = 1
            while True:
                sae = SAE(nwk.sub(0, i))
                top = sae.sub(-1)
                pkw['lrt'] = min(
                    kw.get('lrt', 1e-3) * (len(nwk) - i + 1), 1e-1)

                print('MLP: pre-train sub-SAE top:', top)
                tnr = Tnr(top, xi, xi, **pkw)
                tnr.tune(gdy)

                if i == len(nwk):
                    break

                print('MLP: pre-train sub-SAE all:', sae)
                tnr = Tnr(sae, x, x, **pkw)
                tnr.tune(gdy)

                i = i + 1
                xi = sae.ec(x).eval()
                
        # whole network fine-tuning
        print('train stack:', nwk)
        tr = Tnr(nwk, x, y, u, v, **kw)
        tr.tune(nep)

        return tr

from xnnt.cat import Cat as Cat
from xnnt.pcp import Pcp as Pcp


class AE(Cat):
    """
    Autoencoder
    """

    def __init__(self, fr, **kw):
        """
        Initialize the denosing auto encoder by specifying the the dimension
        of the input  and output.
        The constructor also receives symbolic variables for the weights and
        bias.

        -------- parameters --------
        -- fr: from what to build AE. can a pair of int or a Pcp.
        """
        # the encoder view
        from numpy import number
        if isinstance(fr, Pcp):
            ec = fr
        elif any(isinstance(fr, _) for _ in [tuple, list]) and all(
                isinstance(_, int) for _ in fr):
            ec = Pcp(fr, **kw)
        elif isinstance(fr, number):
            ec = Pcp(fr.astype('<i4').tolist(), **kw)
        else:
            raise ValueError('AE: fr must be int[2] or a Pcp')

        # the decoder view, the weights are tied
        dc = Pcp([ec.dim[1], ec.dim[0]], ec.w.T, **kw)

        # the default view is a concatinated network
        super(AE, self).__init__([ec, dc])
        self.ec = ec
        self.dc = dc


if __name__ == '__main__':
    pass

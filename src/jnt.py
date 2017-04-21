from xnnt.nnt import Nnt
from theano import tensor as T


class Jnt(Nnt):
    """
    Form a parent network by joining two or more child networks.

    The resulting super network takes in a list of inputs {x} of dimension
    [N, d_i], with the 1st dimension indeice the samples.
    The second dimension of x_i must matches the i th. child network.
    The outputs of all child networks are concatenate into one, that is,
    J([N x A], [N x B], ... [N x ...]) -> [N x (A + B + ...)]
    """
    def __init__(self, nts, ops):
        """
        Initialize the network by a list of child builders.

        -------- parameters --------
        nts: child networks builders to be joined.

        ops: operation to perform on child outpus, assuming the 1st dimension
        index the samples.
        The default operation is simply concatenate the outputs
        """
        super(Jnt, self).__init__()

        # input dimension is the list of child input dimenions,
        # output dimension is the sum of child output dimensions.
        dim = [[n.dim[0] for n in nts], sum([n.dim[-1] for n in nts])]

        self.extend(nts)
        self.dim = dim
        self.ops = ops
        self.tag = 'J'

    def __expr__(self, x):
        """
        build symbolic expression of output given the input {x} from child
        inputs being concatenated.
        The higher dimension of x must be equal to the sum of input dimensions
        of all child networks.
        """
        # divide input dimensions
        os, d0 = [], 0
        for n in self:
            os.append(n(x[:, d0:d0 + n.dim[0]]))
            d0 = d0 + n.dim[0]
        if self.ops is None:
            return T.concatenate(os, axis=1)
        else:
            return self.ops(os)


def test_jnt():
    """ test run. """
    from xnnt.pcp import Pcp
    from xnnt.hlp import S
    ns = [Pcp([3, 1]), Pcp([2, 1]), Pcp([3, 2])]
    jn = Jnt(ns)

    import numpy as np
    A = S(np.random.normal(size=[4, 3]), 'A')  # 4 samples, dim = 3
    B = S(np.random.normal(size=[4, 2]), 'B')  # 4 samples, dim = 2
    jo = jn([A, B, A])

    return jo

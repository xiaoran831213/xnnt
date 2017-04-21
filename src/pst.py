try:
    from .nnt import Nnt
except ValueError as e:
    from nnt import Nnt


class Jnt(Nnt):
    """
    Neural networks formed by joining the output of two or more child networks.
    """
    def __init__(self, nts, ops=None):
        """
        Initialize the super neural network by a list of sub networks.

        -------- parameters --------
        nts: child networks to be joined up.
        """
        super(Jnt, self).__init__()

        # first dimension
        dim = [nts[0].dim[0]]

        for p, q in zip(nts[:-1], nts[1:]):
            if p.dim[-1] != q.dim[0]:
                raise Exception('dimension unmatch: {} to {}'.format(p, q))
            dim.append(q.dim[0])

        # last dimension
        dim.append(nts[-1].dim[-1])

        self.extend(nts)
        self.dim = dim

    def __expr__(self, x):
        """
        build symbolic expression of output given input. x is supposdly
        a tensor object.
        """
        for net in self:
            x = net.__expr__(x)
        return x

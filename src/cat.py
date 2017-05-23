from xnnt.nnt import Nnt as Nnt


class Cat(Nnt):
    """
    Neural networks formed by concatinating sub-networks.
    """
    def __init__(self, nts=[]):
        """
        Initialize the super neural network by a list of sub networks.

        -------- parameters --------
        nts: child networks to be chinned up.
        """
        super(Cat, self).__init__()
        self.extend(nts)

    # def extend(self, nts):
    #     """ concatinate a list of networks. """
    #     if not nts:
    #         return None
    #     if len(self) and self.dim[-1] != nts[0].dim[0]:
    #         raise Exception('dimension mismatch:', self, nts[0])
    #     dim = self.dim[:-1]
    #     for p, q in zip(nts[:-1], nts[1:]):
    #         if p.dim[-1] != q.dim[0]:
    #             raise Exception('dimension mismatch:', p, q)
    #         dim.append(q.dim[0])
    #     dim.append(nts[-1].dim[-1])
    #     self.dim = dim
    #     super(Cat, self).extend(nts)

    # def append(self, nnt):
    #     """ concatinate one network. """
    #     if len(self) and self.dim[-1] != nnt.dim[0]:
    #         raise Exception('dimension mismatch:', self, nnt)
    #     self.dim.append(nnt.dim[-1])
    #     super(Cat, self).append(nnt)

    def __expr__(self, x):
        """
        build symbolic expression of output given input. x is supposdly
        a tensor object.
        """
        for net in self:
            x = net.__expr__(x)
        return x

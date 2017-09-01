from xnnt.nnt import Nnt as Nnt


class Cat(Nnt):
    """
    Neural networks formed by concatinating sub-networks.
    """
    def __init__(self, nts):
        """
        Initialize the super neural network by a list of sub networks.

        -------- parameters --------
        nts: child networks to be chinned up.
        """
        # compose chained shape, also perform sanity checking
        dim = [nts[0].dim[0]]
        for u, v in zip(nts[:-1], nts[+1:]):
            if u.dim[-1] == v.dim[0]:
                dim.append(v.dim[0])
                continue
            raise Exception('shape not chainable', u.dim, v.dim)

        # call super initializer
        super(Cat, self).__init__(dim)

        # save the chain
        super(Cat, self).extend(nts)

    def __expr__(self, x):
        """
        build symbolic expression of output given input. x is supposdly
        a tensor object.
        """
        for net in self:
            x = net.__expr__(x)
        return x

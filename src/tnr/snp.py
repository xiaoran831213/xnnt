from copy import deepcopy
from xnnt.hlp import paint
from theano.tensor.sharedvar import TensorSharedVariable as TSV


class Snap(object):
    """
    The component class to enable snap shot for neural network trainers.
    This class must be subclassed along with a Trainer class, to ensure
    the class member 'ep', 'nwk', and the trainer core method 'step' do
    exists.
    """
    __skip__ = ['nwk', '__snap__', '__hist__']

    def __init__(self, *arg, **kwd):
        """ Constructor for class Snap. """
        self.__snap__ = {}      # map of snap shots
        super(Snap, self).__init__(*arg, **kwd)

    def __shot__(self, key=None, **kwd):
        """ take a snap shot. """
        key = -1 if key is None else key
        if '__hist__' in vars(self) and len(self.__hist__) > 0:
            ret = deepcopy(self.__hist__[-1])
        else:
            ret = dict()
        skp = Snap.__skip__
        for k, v in vars(self).items():
            if k in skp or callable(v) or k.startswith('__'):
                continue
            if isinstance(v, TSV):
                ret[k] = v.get_value()
            else:
                ret[k] = deepcopy(v)
        # ret.update(self.__hist__[-1])
        ret['nwk'] = paint(self.nwk)
        self.__snap__[key] = ret
        return ret

    def __rest__(self, key=None, **kwd):
        """ restore a snap shot. """
        key = -1 if key is None else key
        ret, skp = self.__snap__[key], Snap.__skip__
        for k, v in vars(self).items():
            if k in skp or callable(v) or k not in ret:
                continue
            if isinstance(v, TSV):
                v.set_value(ret[k])
            else:
                v = ret[k]
        paint(ret['nwk'], self.nwk)

        # remove history after the snap shot
        if '__hist__' in vars(self) and kwd.get('crop_hist', True):
            ep = ret['ep'].item()
            del self.__hist__[ep + 1:]
        return ret

    def __list__(self, key=None, **kwd):
        """ list snap shots. """
        return self.__snap__.get(key) if key else self.__snap__

    def snap(self, key=None, act='l', **kwd):
        """ List, take, or restore snap shot.
        key: key to identify the snap shot.
        act: the action to take --
        ---- l/0 = list
        ---- s/1 = shot
        ---- r/2 = restore
        
        the function behave differently when given None as the key is given.
        For query action, all shots are returned;
        For shooting action, a default key '-1' is used as the default token
        of the latest status;
        For restoration, '-1' is agained used to load the latest snap shot.
        """
        return {'l': self.__list__, 0: self.__list__,
                's': self.__shot__, 1: self.__shot__,
                'r': self.__rest__, 2: self.__rest__}[act](key, **kwd)
        

import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl
from . import Hypers
from ...libs.tensors import cholesky_robust, debug


class Transport(Hypers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parametrics = []

    def __call__(self, inputs, outputs):
        pass

    def inv(self, inputs, outputs):
        pass

    def logdet_dinv(self, inputs, outputs):
        pass

    def check_hypers(self, parent=''):
        #print('check_hypers', self.parametrics)
        for p in self.parametrics:
            p.check_hypers(parent)

    def default_hypers_dims(self, x=None, y=None):
        r = dict()
        for p in self.parametrics:
            r.update(p.default_hypers_dims(x, y))
        #print(r)
        return r

    def __matmul__(self, other):
        if issubclass(type(other), Transport):
            return TransportComposed(self, other)
        else:
            return TransportComposed(self, other)
    __imatmul__ = __matmul__
    __rmatmul__ = __matmul__


class TransportOperation(Transport):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
        self.op = 'op'

    def check_hypers(self, parent=''):
        self.t1.check_hypers(parent=parent)
        self.t2.check_hypers(parent=parent)
        self.hypers = self.t1.hypers + self.t2.hypers

    def check_dims(self, x=None):
        self.t1.check_dims(x)
        self.t2.check_dims(x)

    def default_hypers_dims(self, x=None, y=None):
        return {**self.t1.default_hypers_dims(x, y), **self.t2.default_hypers_dims(x, y)}

    def __str__(self):
        return str(self.t1) + " "+self.op+" " + str(self.t2)
    __repr__ = __str__


class TransportComposed(TransportOperation):
    def __init__(self, t1: Transport, t2: Transport):
        super().__init__(t1, t2)
        self.op = '@'
        self.name = self.t1.name + " " + self.t2.name

    def __call__(self, t, x):
        return self.t1(t, self.t2(t, x))

    def inv(self, t, y):
        return self.t2.inv(t, self.t1.inv(t, y))

    def logdet_dinv(self, t, y):
        #TODO: Check
        return self.t2.logdet_dinv(t, self.t1.inv(t, y)) + self.t1.logdet_dinv(t, y)


class ID(Transport):
    def __call__(self, inputs, outputs):
        return outputs

    def inv(self, inputs, outputs):
        return outputs

    def logdet_dinv(self, inputs, outputs):
        return tt.ones((), dtype=th.config.floatX)


class TElemwise(Transport):
    pass


class TLinear(Transport):
    pass


class TNoLinear(Transport):
    pass


class TLocation(TElemwise):
    def __init__(self, location = None, x = None, name = None):
        super().__init__(x, name)
        self.location = location
        self.parametrics.append(self.location)

    def __call__(self, inputs, outputs):
        #debug(outputs, name='outputs', force=True)
        #debug(inputs, name='inputs', force=True)
        #debug(self.location(inputs), name='location', force=True)
        return outputs + self.location(inputs)

    def inv(self, inputs, outputs):
        return outputs - self.location(inputs)

    def logdet_dinv(self, inputs, outputs):
        return tt.ones((), dtype=th.config.floatX)


class TScale(TElemwise):
    def __init__(self, scale = None, x = None, name = None):
        super().__init__(x, name)
        self.scale = scale
        self.parametrics.append(self.scale)

    def __call__(self, inputs, outputs):
        return outputs * self.scale(inputs)

    def inv(self, inputs, outputs):
        return outputs / self.scale(inputs)

    def logdet_dinv(self, inputs, outputs):
        return -tt.sum(tt.log(self.scale(inputs)))


class TMapping(TElemwise):
    def __init__(self, mapping = None, x = None, name = None):
        super().__init__(x, name)
        self.mapping = mapping
        self.parametrics.append(self.mapping)

    def __call__(self, inputs, outputs):
        return self.mapping(outputs)

    def inv(self, inputs, outputs):
        return self.mapping.inv(outputs)

    def logdet_dinv(self, inputs, outputs):
        return self.mapping.logdet_dinv(outputs)


class TKernel(TLinear):
    def __init__(self, kernel = None, x = None, name = None):
        super().__init__(x, name)
        self.kernel = kernel
        self.parametrics.append(self.kernel)

    def __call__(self, inputs, outputs):
        cho = cholesky_robust(self.kernel.cov(inputs)).T
        return cho.dot(outputs)

    def inv(self, inputs, outputs):
        cho = cholesky_robust(self.kernel.cov(inputs)).T
        return tsl.solve_lower_triangular(cho, outputs)

    def logdet_dinv(self, inputs, outputs):
        cho = cholesky_robust(self.kernel.cov(inputs)).T
        return - tt.sum(tt.log(tnl.diag(cho)))


class TTriangular(TNoLinear):
    def __init__(self, generator):
        self.generator = generator
        self.parametrics.append(self.generator)

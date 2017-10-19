import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl
from . import Hypers
from .kernels import KernelSum, KernelNoise
from ...libs.tensors import cholesky_robust, debug


class Transport(Hypers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parametrics = []

    def __call__(self, inputs, outputs, noise=False):
        pass

    def diag(self, inputs, outputs, noise=False):
        return self(inputs, outputs, noise=noise)

    def inv(self, inputs, outputs, noise=False):
        pass

    def posterior(self, space, pred, inputs, outputs, noise_pred=False, noise_obs=True, diag=False, inv=False):
        outputs_inv = self.inv(inputs, outputs, noise=True)
        inputs_space = tt.concatenate([inputs, space])
        outputs_space = tt.concatenate([outputs_inv, pred])
        pred_full = self.__call__(inputs_space, outputs_space, noise=True)
        return pred_full[inputs.shape[0]:]

        if not inv:
            outputs_inv = self.inv(inputs, outputs, noise=noise_obs)
            inputs_space = tt.concatenate([inputs, space])
            outputs_space = tt.concatenate([outputs_inv, pred])
            if diag:
                pred_full = self.diag(inputs_space, outputs_space, noise=noise_pred)
            else:
                pred_full = self.__call__(inputs_space, outputs_space, noise=noise_pred)
        else:
            inputs_space = tt.concatenate([inputs, space])
            outputs_space = tt.concatenate([outputs, pred])
            pred_full = self.inv(inputs_space, outputs_space, noise=noise_obs)
        return pred_full[inputs.shape[0]:]

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
    #__imatmul__ = __matmul__
    #__rmatmul__ = __matmul__


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

    def __call__(self, inputs, outputs, noise=False):
        return self.t1(inputs, self.t2(inputs, outputs, noise=noise), noise=noise)

    def diag(self, inputs, outputs, noise=False):
        return self.t1.diag(inputs, self.t2(inputs, outputs, noise=noise), noise=noise)

    def inv(self, inputs, outputs, noise=False):
        return self.t2.inv(inputs, self.t1.inv(inputs, outputs, noise=noise), noise=noise)

    def logdet_dinv(self, inputs, outputs):
        return self.t2.logdet_dinv(inputs, self.t1.inv(inputs, outputs, noise=True)) + self.t1.logdet_dinv(inputs, outputs)

    def posterior(self, space, pred, inputs, outputs, noise_pred=False, noise_obs=True, diag=False, inv=False):
        outputs_inv = self.inv(inputs, outputs, noise=True)
        inputs_space = tt.concatenate([inputs, space])
        outputs_space = tt.concatenate([outputs_inv, pred])
        pred_full = self.__call__(inputs_space, outputs_space, noise=True)
        return pred_full[inputs.shape[0]:]

class ID(Transport):
    def __call__(self, inputs, outputs, noise=False):
        return outputs

    def inv(self, inputs, outputs, noise=False):
        return outputs

    def logdet_dinv(self, inputs, outputs):
        return tt.ones((), dtype=th.config.floatX)


class TElemwise(Transport):
    def posterior(self, space, pred, inputs=None, outputs=None, noise_pred=False, noise_obs=True, diag=False, inv=False):
        return self(space, pred, noise=noise_pred)


class TLinear(Transport):
    pass


class TNoLinear(Transport):
    pass


class TLocation(TElemwise):
    def __init__(self, location = None, x = None, name = None):
        super().__init__(x, name)
        self.location = location
        self.parametrics.append(self.location)

    def __call__(self, inputs, outputs, noise=False):
        #debug(outputs, name='outputs', force=True)
        #debug(inputs, name='inputs', force=True)
        #debug(self.location(inputs), name='location', force=True)
        return outputs + self.location(inputs)

    def inv(self, inputs, outputs, noise=False):
        return outputs - self.location(inputs)

    def logdet_dinv(self, inputs, outputs):
        return tt.zeros((), dtype=th.config.floatX)


class TScale(TElemwise):
    def __init__(self, scale = None, x = None, name = None):
        super().__init__(x, name)
        self.scale = scale
        self.parametrics.append(self.scale)

    def __call__(self, inputs, outputs, noise=False):
        return outputs * self.scale(inputs)

    def inv(self, inputs, outputs, noise=False):
        return outputs / self.scale(inputs)

    def logdet_dinv(self, inputs, outputs):
        _scale = debug(self.scale(inputs), 'scale', force=False)
        _log = debug(tt.log(_scale), 'log', force=False)
        _sum = debug(tt.sum(_log), 'log', force=False)
        return -_sum


class TMapping(TElemwise):
    def __init__(self, mapping = None, x = None, name = None):
        super().__init__(x, name)
        self.mapping = mapping
        self.parametrics.append(self.mapping)

    def __call__(self, inputs, outputs, noise=False):
        return self.mapping(outputs)

    def inv(self, inputs, outputs, noise=False):
        return self.mapping.inv(outputs)

    def logdet_dinv(self, inputs, outputs):
        return self.mapping.logdet_dinv(outputs)


class TKernel(TLinear):
    def __init__(self, kernel, noisy=False, x = None, name = None):
        super().__init__(x, name)
        if noisy:
            self.kernel = kernel
            self.noisy = KernelSum(self.kernel, KernelNoise(name='Noise'+kernel.name))

        else:
            self.kernel = kernel
            self.noisy = kernel
        self.parametrics.append(self.noisy)

    def __call__(self, inputs, outputs, noise=False):
        # cho = cholesky_robust(tnl.diag(tnl.diag(self.kernel.cov(inputs))))
        if noise:
            cho = cholesky_robust(self.noisy.cov(inputs))
        else:
            cho = cholesky_robust(self.kernel.cov(inputs))
        return cho.dot(outputs)

    def diag(self, inputs, outputs, noise=False):
        if noise:
            cho = tt.sqrt(tnl.diag(self.noisy.cov(inputs)))
            #cho = cholesky_robust(self.noisy.cov(inputs))
        else:
            cho = tt.sqrt(tnl.diag(self.kernel.cov(inputs)))
            #cho = cholesky_robust(self.kernel.cov(inputs))
        return cho * outputs

    def inv(self, inputs, outputs, noise=False):
        if noise:
            cho = cholesky_robust(self.noisy.cov(inputs))
        else:
            cho = cholesky_robust(self.kernel.cov(inputs))
        return tsl.solve(cho.T, outputs)

    def logdet_dinv(self, inputs, outputs):
        cho = cholesky_robust(self.noisy.cov(inputs))
        return - tt.sum(tt.log(tnl.diag(cho)))

    def posterior(self, space, pred, inputs, outputs, noise_pred=False, noise_obs=True, diag=False, inv=False):
        outputs_inv = self.inv(inputs, outputs, noise=True)
        inputs_space = tt.concatenate([inputs, space])
        outputs_space = tt.concatenate([outputs_inv, pred])
        pred_full = self.__call__(inputs_space, outputs_space, noise=False)
        return pred_full[inputs.shape[0]:]


class TTriangular(TNoLinear):
    def __init__(self, generator):
        self.generator = generator
        self.parametrics.append(self.generator)

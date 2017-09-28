import numpy as np
import pymc3 as pm
import theano as th
import theano.tensor as tt
import theano.tensor.nlinalg as tnl
from .stochastic import StochasticProcess
from .hypers import Hypers
from ..libs import DictObj
#import types
#from .elliptical import debug_p


class Density(Hypers):

    def __init__(self, **kwargs):
        self.locations = kwargs

    def check_dims(self, *args, **kwargs):
        _ = {l.check_dims(*args, **kwargs) for k, l in self.locations.items()}

    def check_hypers(self, *args, **kwargs):
        _ = {l.check_hypers(*args, **kwargs) for k, l in self.locations.items()}

    def check_potential(self, *args, **kwargs):
        _ = {l.check_potential(*args, **kwargs) for k, l in self.locations.items()}

    def default_hypers_dims(self, *args, **kwargs):
        return {l.default_hypers_dims(*args, **kwargs) for k, l in self.locations.items()}

    def distribution(self, name, inputs, outputs, testval, dtype):
        pass

    def median(self, space):
        pass

    def mean(self, space):
        pass

    def mode(self, space):
        pass

    def variance(self, space):
        return tt.pow(self.std(space), 2)

    def std(self, space):
        return tt.sqrt(self.variance(space))


class SymmetricDensity(Hypers):

    def median(self, *args, **kwargs):
        return self.mean(*args, **kwargs)

    def mode(self, *args, **kwargs):
        return self.mean(*args, **kwargs)


class Uniform(SymmetricDensity):

    def __init__(self, lower, upper):
        self.locations = DictObj()
        self.locations['lower'] = lower
        self.locations['upper'] = upper

    def mean(self, space):
        return np.float32(0.5)*(self.locations.lower(space)+self.locations.upper(space))

    def variance(self, space):
        return np.float32(1/12)*(self.locations.upper(space)-self.locations.lower(space))**2

    def lower(self, space):
        return self.locations.lower(space)

    def upper(self, space):
        return self.locations.upper(space)

    def distribution(self, name, inputs, outputs, testval, dtype):
        return pm.Uniform(name=name, observed=outputs, testval=testval, dtype=dtype,
                          lower=self.lower(inputs), upper=self.upper(inputs))


class Normal(SymmetricDensity):

    def __init__(self, mu, sigma):
        self.locations = DictObj()
        self.locations['mu'] = mu
        self.locations['sigma'] = sigma

    def mean(self, space):
        return self.locations.mu(space)

    def std(self, space):
        return self.locations.sigma(space)

    def distribution(self, name, inputs, outputs, testval, dtype):
        return pm.Normal(name=name, observed=outputs, testval=testval, dtype=dtype,
                         mu=self.mean(inputs), sd=self.std(inputs))


class StudentT(Normal):

    def __init__(self, mu, sigma, nu):
        self.locations = DictObj()
        self.locations['mu'] = mu
        self.locations['sigma'] = sigma
        self.locations['nu'] = nu

    def distribution(self, name, inputs, outputs, testval, dtype):
        return pm.StudentT(name=name, observed=outputs, testval=testval, dtype=dtype,
                           mu=self.mean(inputs), sd=self.std(inputs), nu=self.locations.nu(inputs))


class MarginalProcess(StochasticProcess):
    def __init__(self, space=None, density: Density=None, *args, **kwargs):
        self.f_density = density
        kwargs['space'] = space
        super().__init__(*args, **kwargs)

    def _check_hypers(self):
        self.f_density.check_dims(self.inputs)
        self.f_density.check_hypers(self.name + '_')
        self.f_density.check_potential()

    def default_hypers(self):
        x = self.inputs
        y = self.outputs
        return self.f_density.default_hypers_dims(x, y)

    def th_define_process(self):
        super().th_define_process()
        self.distribution = self.f_density.distribution(self.name, inputs=self.th_inputs, outputs=self.th_outputs,
                                                        testval=self.outputs, dtype=th.config.floatX)

    def th_median(self, prior=False, noise=False):
        return self.f_density.median(self.th_space)

    def th_mean(self, prior=False, noise=False):
        return self.f_density.mean(self.th_space)

    def th_variance(self, prior=False, noise=False):
        return self.f_density.variance(self.th_space)

    def th_covariance(self, prior=False, noise=False):
        return tnl.diag(self.f_density.variance(self.th_space))





import types
import numpy as np
import pymc3 as pm
import theano as th
import theano.tensor as tt
import theano.tensor.nlinalg as tnl
from .stochastic import StochasticProcess
from .hypers import Hypers
from ..libs import DictObj
from scipy import stats
#import types
#from .elliptical import debug_p


class Density:

    def __init__(self, **kwargs):
        self.locations = kwargs

    def check_dims(self, *args, **kwargs):
        _ = {l.check_dims(*args, **kwargs) for k, l in self.locations.items()}

    def check_hypers(self, *args, **kwargs):
        _ = {l.check_hypers(*args, **kwargs) for k, l in self.locations.items()}

    def check_potential(self, *args, **kwargs):
        _ = {l.check_potential(*args, **kwargs) for k, l in self.locations.items()}

    def default_hypers_dims(self, *args, **kwargs):
        r = DictObj()
        for k, l in self.locations.items():
            r.update(l.default_hypers_dims(*args, **kwargs))
        return r

    def distribution(self, name, inputs, outputs, testval, dtype):
        pass

    def th_median(self, space):
        pass

    def th_mean(self, space):
        pass

    def th_mode(self, space):
        pass

    def th_variance(self, space):
        return tt.pow(self.th_std(space), 2)

    def th_std(self, space):
        return tt.sqrt(self.th_variance(space))


class MarginalProcess(StochasticProcess):
    def __init__(self, space=None, density: Density=None, *args, **kwargs):
        self.f_density = density
        kwargs['space'] = space
        if 'name' not in kwargs:
            kwargs['name'] = 'MP'
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

    def _compile_methods(self):
        self.lower = types.MethodType(self._method_name('th_lower'), self)
        self.upper = types.MethodType(self._method_name('th_upper'), self)
        self.freedom = types.MethodType(self._method_name('th_freedom'), self)
        super()._compile_methods()

    def th_lower(self, prior=False, noise=False):
        return self.f_density.th_lower(self.th_space)

    def th_upper(self, prior=False, noise=False):
        return self.f_density.th_upper(self.th_space)

    def th_freedom(self, prior=False, noise=False):
        return self.f_density.th_freedom(self.th_space)

    def th_median(self, prior=False, noise=False):
        return self.f_density.th_median(self.th_space)

    def th_mean(self, prior=False, noise=False):
        return self.f_density.th_mean(self.th_space)

    def th_variance(self, prior=False, noise=False):
        return self.f_density.th_variance(self.th_space)

    def th_covariance(self, prior=False, noise=False):
        return tnl.diag(self.f_density.th_variance(self.th_space))

    def quantiler(self, params=None, space=None, inputs=None, outputs=None, q=0.975, prior=False, noise=False):
        if space is None:
            space = self.space
        if isinstance(self.f_density, StudentT):
            nu = self.freedom(params, space, inputs, outputs, prior=prior, noise=noise),
            ppf = stats.t.ppf(q, nu,
                              loc=self.mean(params, space, inputs, outputs, prior=prior, noise=noise),
                              scale=self.std(params, space, inputs, outputs, prior=prior, noise=noise))
        elif isinstance(self.f_density, Uniform):
            lower = self.lower(params, space, inputs, outputs, prior=prior, noise=noise)
            upper = self.upper(params, space, inputs, outputs, prior=prior, noise=noise)
            ppf = stats.uniform.ppf(q, loc=lower, scale=upper - lower)
        else:
            ppf = stats.norm.ppf(q, loc=self.mean(params, space, inputs, outputs, prior=prior, noise=noise),
                                 scale=self.std(params, space, inputs, outputs, prior=prior, noise=noise))
        return ppf

    def sampler(self, params=None, space=None, inputs=None, outputs=None, samples=1, prior=False, noise=False):
        if space is None:
            space = self.space
        if isinstance(self.f_density, StudentT):
            nu = self.freedom(params, space, inputs, outputs, prior=prior, noise=noise),
            rand = stats.t.rvs(nu,
                               loc=self.mean(params, space, inputs, outputs, prior=prior, noise=noise),
                               scale=self.std(params, space, inputs, outputs, prior=prior, noise=noise),
                               size=(samples, len(space)))
        elif isinstance(self.f_density, Uniform):
            lower = self.lower(params, space, inputs, outputs, prior=prior, noise=noise)
            upper = self.upper(params, space, inputs, outputs, prior=prior, noise=noise)
            rand = stats.uniform.rvs(loc=lower, scale=upper - lower, size=(samples, len(space)))
        else:
            rand = stats.norm.rvs(loc=self.mean(params, space, inputs, outputs, prior=prior, noise=noise),
                                  scale=self.std(params, space, inputs, outputs, prior=prior, noise=noise),
                                  size=(samples, len(space)))
        return rand.T


class SymmetricDensity(Density):

    def th_median(self, *args, **kwargs):
        return self.th_mean(*args, **kwargs)

    def th_mode(self, *args, **kwargs):
        return self.th_mean(*args, **kwargs)


class Uniform(SymmetricDensity):

    def __init__(self, lower, upper):
        self.locations = DictObj()
        self.locations['lower'] = lower
        self.locations['upper'] = upper

    def th_mean(self, space):
        return np.float32(0.5)*(self.locations.lower(space)+self.locations.upper(space))

    def th_variance(self, space):
        return np.float32(1/12)*(self.locations.upper(space)-self.locations.lower(space))**2

    def th_lower(self, space):
        return self.locations.lower(space)

    def th_upper(self, space):
        return self.locations.upper(space)

    def distribution(self, name, inputs, outputs, testval, dtype):
        return pm.Uniform(name=name, observed=outputs, testval=testval, dtype=dtype,
                          lower=self.th_lower(inputs), upper=self.th_upper(inputs))


class Normal(SymmetricDensity):

    def __init__(self, mu, sigma):
        self.locations = DictObj()
        self.locations['mu'] = mu
        self.locations['sigma'] = sigma

    def th_mean(self, space):
        return self.locations.mu(space)

    def th_std(self, space):
        return tt.abs_(self.locations.sigma(space))

    def distribution(self, name, inputs, outputs, testval, dtype):
        return pm.Normal(name=name, observed=outputs, testval=testval, dtype=dtype,
                         mu=self.th_mean(inputs), sd=self.th_std(inputs))


class StudentT(Normal):

    def __init__(self, mu, sigma, nu):
        self.locations = DictObj()
        self.locations['mu'] = mu
        self.locations['sigma'] = sigma
        self.locations['nu'] = nu

    def th_freedom(self, space):
        return self.locations.nu(space)

    def distribution(self, name, inputs, outputs, testval, dtype):
        return pm.StudentT(name=name, observed=outputs, testval=testval, dtype=dtype,
                           mu=self.th_mean(inputs), sd=self.th_std(inputs), nu=self.th_freedom(inputs))







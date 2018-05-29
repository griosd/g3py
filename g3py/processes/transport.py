import types
import numpy as np
import scipy as sp
import pymc3 as pm
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl
from scipy import stats
from theano.ifelse import ifelse
from .elliptical import debug_p
from .stochastic import StochasticProcess
from .hypers.transports import Transport, ID
from ..libs.tensors import debug


class TransportProcess(StochasticProcess):
    def __init__(self, space=None, transport: Transport=ID(), *args, **kwargs):
        #print('TransportProcess')
        self.f_transport = transport
        kwargs['space'] = space
        super().__init__(*args, **kwargs)

    def _check_hypers(self):
        self.f_transport.check_dims(self.inputs)
        self.f_transport.check_hypers(self.name + '_')
        self.f_transport.check_potential()

    def default_hypers(self):
        x = self.inputs
        y = self.outputs
        return self.f_transport.default_hypers_dims(x, y)

    def th_define_process(self):
        #print('stochastic_define_process')
        # Basic Tensors
        self.prior_transport = self.f_transport(self.th_space, self.th_vector, noise=False)
        self.prior_transport_noise = self.f_transport(self.th_space, self.th_vector, noise=True)

        #print('posterior_transport')
        self.posterior_transport = self.f_transport.posterior(self.th_space, self.th_vector, self.th_inputs, self.th_outputs, noise_pred=False, noise_obs=True)
        #self.posterior_transport = value = debug(self.posterior_transport, 'posterior_transport', force=True)

        #print('posterior_transport_noise')
        self.posterior_transport_noise = self.f_transport.posterior(self.th_space, self.th_vector, self.th_inputs, self.th_outputs, noise_pred=True, noise_obs=True)


        self.diag_prior_transport = self.f_transport.diag(self.th_space, self.th_vector, noise=False)
        self.diag_prior_transport_noise = self.f_transport.diag(self.th_space, self.th_vector, noise=True)

        #print('diag_posterior_transport')
        self.diag_posterior_transport = self.f_transport.posterior(self.th_space, self.th_vector, self.th_inputs, self.th_outputs, noise_pred=False, noise_obs=True, diag=True)
        #print('diag_posterior_transport_noise')
        self.diag_posterior_transport_noise = self.f_transport.posterior(self.th_space, self.th_vector, self.th_inputs, self.th_outputs, noise_pred=True, noise_obs=True, diag=True)



        self.inv_prior_transport = self.f_transport.inv(self.th_space, self.th_vector, noise=False)
        self.inv_prior_transport_noise = self.f_transport.inv(self.th_space, self.th_vector, noise=True)
        #print('inv_posterior_transport')
        self.inv_posterior_transport = self.f_transport.posterior(self.th_space, self.th_vector, self.th_inputs, self.th_outputs, noise_pred=False, noise_obs=True, inv=True)
        #print('inv_posterior_transport_noise')
        self.inv_posterior_transport_noise = self.f_transport.posterior(self.th_space, self.th_vector, self.th_inputs, self.th_outputs, noise_pred=True, noise_obs=True, inv=True)




    def th_transport(self, prior=False, noise=False):
        if prior:
            if noise:
                return self.prior_transport_noise
            else:
                return self.prior_transport
        else:
            if noise:
                return self.posterior_transport_noise
            else:
                return self.posterior_transport

    def th_transport_diag(self, prior=False, noise=False):
        if prior:
            if noise:
                return self.diag_prior_transport_noise
            else:
                return self.diag_prior_transport
        else:
            if noise:
                return self.diag_posterior_transport_noise
            else:
                return self.diag_posterior_transport

    def th_transport_inv(self, prior=False, noise=False):
        if prior:
            if noise:
                return self.inv_prior_transport_noise
            else:
                return self.inv_prior_transport
        else:
            if noise:
                return self.inv_posterior_transport_noise
            else:
                return self.inv_posterior_transport

    def th_median(self, prior=False, noise=False, simulations=None):
        pass

    def th_mean(self, prior=False, noise=False, simulations=None):
        pass

    def th_variance(self, prior=False, noise=False):
        pass

    def th_covariance(self, prior=False, noise=False):
        pass

    def _compile_methods(self, *args, **kwargs):
        super()._compile_methods(*args, **kwargs)
        self.transport = types.MethodType(self._method_name('th_transport'), self)
        self.transport_diag = types.MethodType(self._method_name('th_transport_diag'), self)
        self.transport_inv = types.MethodType(self._method_name('th_transport_inv'), self)

    def plot_transport(self, params=None, space=None, inputs=None, prior=True, noise=False, centers=[1/10, 1/2, 9/10]):
        pass

    def plot_model(self, params=None, indexs=None, kernel=True, mapping=True, marginals=True, bivariate=True):
        pass

    def plot_distribution(self, index=0, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, quantiles_noise=False, noise=False, prior=False, sigma=4, neval=100, title=None, swap=False, label=None):
        pass

    def plot_distribution2D(self, indexs=[0, 1], params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, quantiles_noise=False, noise=False, prior=False, sigma_1=2, sigma_2=2, neval=33, title=None):
        pass


class TransportGaussianProcess(TransportProcess):

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'TGP'
        super().__init__(*args, **kwargs)

    def th_define_process(self):
        super().th_define_process()
        self.distribution = TransportGaussianDistribution(self.name, transport=self.f_transport,
                                                          observed=self.th_outputs, inputs=self.th_inputs,
                                                          testval=self.outputs, dtype=th.config.floatX)

    def th_mean(self, prior=False, noise=False, simulations=None, n=10):
        #debug_p('mean')
        #_a, _w = np.polynomial.hermite.hermgauss(n)
        #a = th.shared(_a.astype(th.config.floatX), borrow=False).dimshuffle([0, 'x'])
        #w = th.shared(_w.astype(th.config.floatX), borrow=False)
        #return self.transport_gauss_hermite(lambda i, v: self.f_transport(i, v), self.th_space, a, w)
        pass

    def th_variance(self, prior=False, noise=False, n=10):
        #debug_p('variance')
        #_a, _w = np.polynomial.hermite.hermgauss(n)
        #a = th.shared(_a.astype(th.config.floatX), borrow=False).dimshuffle([0, 'x'])
        #w = th.shared(_w.astype(th.config.floatX), borrow=False)
        #return self.transport_gauss_hermite(lambda i, v: self.f_transport(i, v)**2, self.th_space, a, w) \
        #       - self.th_mean(prior=prior, noise=noise) ** 2
        pass

    def th_covariance(self, prior=False, noise=False):
        pass

    @classmethod
    def transport_gauss_hermite(cls, f, i, a, w):
        grille = np.sqrt(2).astype(th.config.floatX) * a
        return tt.dot(w, f(i, grille.flatten()).reshape(grille.shape)) / np.sqrt(np.pi).astype(th.config.floatX)

    def mean(self, params=None, space=None, inputs=None, outputs=None, prior=False, noise=False, simulations=None):
        if simulations is None:
            simulations = 30

        if type(simulations) is int:
            samples = self.sampler(params=params, space=space, inputs=inputs, outputs=outputs, samples=simulations,
                                   prior=prior, noise=noise)
        else:
            samples = simulations
        return samples.mean(axis=1)

    def std(self, params=None, space=None, inputs=None, outputs=None, prior=False, noise=False, simulations=None):
        if simulations is None:
            simulations = 30

        if type(simulations) is int:
            samples = self.sampler(params=params, space=space, inputs=inputs, outputs=outputs, samples=simulations,
                                   prior=prior, noise=noise)
        else:
            samples = simulations
        return samples.std(axis=1)

    def quantiler(self, params=None, space=None, inputs=None, outputs=None, q=0.975, prior=False, noise=False, simulations=None):
        if simulations is None:
            simulations = 30

        if type(simulations) is int:
            samples = self.sampler(params=params, space=space, inputs=inputs, outputs=outputs, samples=simulations,
                     prior=prior, noise=noise)
        else:
            samples = simulations
        return np.nanpercentile(samples, 100*q, axis=1)

    def sampler(self, params=None, space=None, inputs=None, outputs=None, samples=1, prior=False, noise=False):
        if space is None:
            space = self.space
        rand = np.random.randn(len(space), samples)
        return np.array([self.transport(params, space, inputs, outputs, vector=rand[:, i],
                              prior=prior, noise=noise).T for i in range(samples)]).T


class TransportGaussianDistribution(pm.Continuous):
    def __init__(self, transport=ID(), inputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transport = transport
        self.th_inputs = inputs

    @classmethod
    def logp_t(cls, value, transport, inputs):
        #print(value.tag.test_value)
        #print(mu.tag.test_value)
        #print(mapping.inv(value).tag.test_value)

        value = debug(value, 'value', force=False)
        delta = transport.inv(inputs, value, noise=True)
        det_m = transport.logdet_dinv(inputs, value)
        delta = debug(delta, 'delta', force=False)

        npi = np.float32(-0.5) * value.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
        dot2 = np.float32(-0.5) * delta.dot(delta.T)

        npi = debug(npi, 'npi', force=False)
        dot2 = debug(dot2, 'dot2', force=False)
        det_m = debug(det_m, 'det_m', force=False)

        r = npi + dot2 + det_m

        cond1 = tt.or_(tt.any(tt.isinf_(delta)), tt.any(tt.isnan_(delta)))
        cond2 = tt.or_(tt.any(tt.isinf_(det_m)), tt.any(tt.isnan_(det_m)))

        return ifelse(cond1, np.float32(-1e30), ifelse(cond2, np.float32(-1e30), r))

    def logp(self, value):
        return self.logp_t(value, self.transport, self.th_inputs)

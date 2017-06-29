import numpy as np
import scipy as sp
import pymc3 as pm
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl
from scipy import stats
from theano.ifelse import ifelse
from .stochastic import EllipticalProcess
from .hypers.mappings import Identity
from ..libs.tensors import cholesky_robust, debug#, tt_to_num


class GaussianProcess(EllipticalProcess):

    def _define_process(self):
        super()._define_process()
        self.distribution = TransformedGaussianDistribution(self.name, mu=self.prior_location_inputs,
                                                            cov=self.prior_kernel_inputs, mapping=Identity(),
                                                            observed=self.th_outputs, testval=self.th_outputs,
                                                            dtype=th.config.floatX)

    def median(self, prior=False, noise=False):
        debug('median')
        if prior:
            latent = self.prior_location_space
        else:
            latent = self.posterior_location_space
        return self.mapping(latent)

    def mean(self, prior=False, noise=False):
        debug('mean')
        if prior:
            latent = self.prior_location_space
        else:
            latent = self.posterior_location_space
        return self.mapping(latent)

    def variance(self, prior=False, noise=False):
        debug('variance')
        if prior:
            if noise:
                return tnl.extract_diag(self.prior_kernel_space)
            else:
                return tnl.extract_diag(self.prior_kernel_f_space)
        else:
            if noise:
                return tnl.extract_diag(self.posterior_kernel_space)
            else:
                return tnl.extract_diag(self.posterior_kernel_f_space)

    def quantiler(self, q=0.975, prior=False, noise=False):
        debug('quantiler')
        p = stats.norm.ppf(q)
        return self.mapping(self.mean(prior=prior, noise=noise) + p * self.std(prior=prior, noise=noise))

    def sampler(self, nsamples=1, prior=False, noise=False):
        debug('sampler')
        rand = np.random.randn(len(self.th_space), nsamples)
        return self.mapping(self.mean(prior=prior, noise=noise) + self.cholesky(prior=prior, noise=noise).dot(rand))


def debug(*args, **kwargs):
    print(*args, **kwargs)


class TransformedGaussianProcess(GaussianProcess):

    def _define_process(self):
        super()._define_process()

        self.distribution = TransformedGaussianDistribution(self.name, mu=self.prior_location_inputs,
                                                            cov=self.prior_kernel_inputs, mapping=self.mapping,
                                                            observed=self.th_outputs, testval=self.th_outputs,
                                                            dtype=th.config.floatX)

    def mean(self, prior=False, noise=False, n=10):
        debug('mean')
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX), borrow=True).dimshuffle([0, 'x'])
        w = th.shared(_w.astype(th.config.floatX), borrow=True)
        if prior:
            return self.gauss_hermite(lambda v: self.mapping(v) , self.prior_location_space,
                                      self.std(prior=prior, noise=noise), a, w)
        else:
            return self.gauss_hermite(lambda v: self.mapping(v) , self.posterior_location_space,
                                      self.std(prior=prior, noise=noise), a, w)

    def variance(self, prior=False, noise=False, n=10):
        debug('mean')
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX), borrow=True).dimshuffle([0, 'x'])
        w = th.shared(_w.astype(th.config.floatX), borrow=True)
        if prior:
            return self.gauss_hermite(lambda v: self.mapping(v) ** 2, self.prior_location_space,
                                      self.std(prior=prior, noise=noise), a, w) - self.mean(prior=prior, noise=noise) ** 2
        else:
            return self.gauss_hermite(lambda v: self.mapping(v) ** 2, self.posterior_location_space,
                                      self.std(prior=prior, noise=noise), a, w) - self.mean(prior=prior, noise=noise) ** 2

    @classmethod
    def gauss_hermite(cls, f, mu, sigma, a, w):
        grille = mu + sigma * np.sqrt(2).astype(th.config.floatX) * a
        return tt.dot(w, f(grille.flatten()).reshape(grille.shape)) / np.sqrt(np.pi).astype(th.config.floatX)


class TransformedGaussianDistribution(pm.Continuous):
    def __init__(self, mu, cov, mapping=Identity(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov
        self.mapping = mapping

    @classmethod
    def logp_cho(cls, value, mu, cho, mapping):

        delta = mapping.inv(value) - mu
        lcho = tsl.solve_lower_triangular(cho, delta)

        npi = np.float32(-0.5) * cho.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
        dot2 = np.float32(-0.5) * lcho.T.dot(lcho)
        det_k = - tt.sum(tt.log(tnl.diag(cho)))
        det_m = mapping.logdet_dinv(value)
        r = npi + dot2 + det_k + det_m

        cond1 = tt.or_(tt.any(tt.isinf_(delta)), tt.any(tt.isnan_(delta)))
        cond2 = tt.or_(tt.any(tt.isinf_(det_m)), tt.any(tt.isnan_(det_m)))
        cond3 = tt.or_(tt.any(tt.isinf_(lcho)), tt.any(tt.isnan_(lcho)))
        return ifelse(cond1, np.float32(-1e30), ifelse(cond2, np.float32(-1e30), ifelse(cond3, np.float32(-1e30), r)))

    def logp(self, value):
        return self.logp_cho(value, self.mu, self.cho, self.mapping)

    @property
    def cho(self):
        try:
            return cholesky_robust(self.cov) #tt_to_num
        except:
            raise sp.linalg.LinAlgError("not cholesky")
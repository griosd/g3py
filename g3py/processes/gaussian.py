import numpy as np
import scipy as sp
import pymc3 as pm
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl
from scipy import stats
from theano.ifelse import ifelse
from .elliptical import EllipticalProcess, debug_p
from .hypers.mappings import Identity
from ..libs.tensors import cholesky_robust, debug, tt_to_bounded, tt_eval


class GaussianProcess(EllipticalProcess):

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'GP'
        super().__init__(*args, **kwargs)

    def th_define_process(self):
        #print('gaussian_define_process')
        super().th_define_process()
        self.distribution = WarpedGaussianDistribution(self.name, mu=self.prior_location_inputs,
                                                       cov=self.prior_kernel_inputs, mapping=self.f_mapping,
                                                       observed=self.th_outputs, testval=self.outputs,
                                                       dtype=th.config.floatX)

    def th_logpredictive(self, prior=False, noise=False):
        return WarpedGaussianDistribution.logp_cho(value=self.th_vector,
                                                   mu=self.th_location(prior=prior, noise=noise),
                                                   cho=self.th_kernel_sd(prior=prior, noise=True, cholesky=True),
                                                   mapping=self.f_mapping)

    def quantiler(self, params=None, space=None, inputs=None, outputs=None, q=0.975, prior=False, noise=False):
        #debug_p('quantiler' + str(q) + str(prior) + str(noise))
        p = stats.norm.ppf(q)
        gp_quantiler = self.location(params, space, inputs, outputs, prior=prior, noise=noise) + p*self.kernel_sd(params, space, inputs, outputs, prior=prior, noise=noise)
        return self.mapping(params, space, inputs, outputs=gp_quantiler) #self.f_mapping

    def sampler(self, params=None, space=None, inputs=None, outputs=None, samples=1, prior=False, noise=False):
        #debug_p('sampler' + str(samples) + str(prior) + str(noise)+str(len(self.space)))
        if space is None:
            space = self.space
        rand = np.random.randn(len(space), samples)
        qp_samples = self.location(params, space, inputs, outputs, prior=prior, noise=noise)[:, None] + \
                     self.cholesky(params, space, inputs, outputs, prior=prior, noise=noise).dot(rand)
        return np.array([self.mapping(params, space, inputs, outputs=k.T) for k in qp_samples.T]).T

    def th_cross_mean(self, prior=False, noise=False, cross_kernel=None):
        if prior:
            return self.prior_location_space
        if cross_kernel is None:
            cross_kernel = self.f_kernel
        return self.prior_location_space + cross_kernel.cov(self.th_space_, self.th_inputs_).dot(
            tsl.solve(self.prior_kernel_inputs, self.mapping_outputs - self.prior_location_inputs))


class WarpedGaussianProcess(GaussianProcess):

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'WGP'
        super().__init__(*args, **kwargs)

    def th_mean(self, prior=False, noise=False, simulations=None, n=10):
        debug_p('mean')
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX), borrow=False).dimshuffle([0, 'x'])
        w = th.shared(_w.astype(th.config.floatX), borrow=False)
        return self.gauss_hermite(lambda v: self.f_mapping(v), self.th_location(prior=prior, noise=noise),
                                  self.th_kernel_sd(prior=prior, noise=noise), a, w)

    def th_variance(self, prior=False, noise=False, n=10):
        debug_p('variance')
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX), borrow=False).dimshuffle([0, 'x'])
        w = th.shared(_w.astype(th.config.floatX), borrow=False)
        return self.gauss_hermite(lambda v: self.f_mapping(v) ** 2, self.th_location(prior=prior, noise=noise),
                                  self.th_kernel_sd(prior=prior, noise=noise), a, w) - self.th_mean(prior=prior, noise=noise) ** 2

    def th_covariance(self, prior=False, noise=False):
        pass

    @classmethod
    def gauss_hermite(cls, f, mu, sigma, a, w):
        grille = mu + sigma * np.sqrt(2).astype(th.config.floatX) * a
        return tt.dot(w, f(grille.flatten()).reshape(grille.shape)) / np.sqrt(np.pi).astype(th.config.floatX)


class WarpedGaussianDistribution(pm.Continuous):
    def __init__(self, mu, cov, mapping=Identity(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov
        self.mapping = mapping

    @classmethod
    def logp_cho(cls, value, mu, cho, mapping):
        #print(value.tag.test_value)
        #print(mu.tag.test_value)
        #print(mapping.inv(value).tag.test_value)
        #mu = debug(mu, 'mu', force=True)

        value = debug(value, 'value', force=False)
        delta = mapping.inv(value) - mu

        #delta = debug(delta, 'delta', force=True)
        cho = debug(cho, 'cho', force=False)
        lcho = tsl.solve_lower_triangular(cho.T, delta)
        lcho = debug(lcho, 'lcho', force=False)

        lcho2 = lcho.T.dot(lcho)
        #lcho2 = debug(lcho2, 'lcho2', force=True)

        npi = np.float32(-0.5) * cho.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
        dot2 = np.float32(-0.5) * lcho2

        #diag = debug(tnl.diag(cho), 'diag', force=True)
        #_log= debug(tt.log(diag), 'log', force=True)

        det_k = - tt.sum(tt.log(tnl.diag(cho)))
        det_m = mapping.logdet_dinv(value)

        npi = debug(npi, 'npi', force=False)
        dot2 = debug(dot2, 'dot2', force=False)
        det_k = debug(det_k, 'det_k', force=False)
        det_m = debug(det_m, 'det_m', force=False)

        r = npi + dot2 + det_k + det_m

        cond1 = tt.or_(tt.any(tt.isinf_(delta)), tt.any(tt.isnan_(delta)))
        cond2 = tt.or_(tt.any(tt.isinf_(det_m)), tt.any(tt.isnan_(det_m)))
        cond3 = tt.or_(tt.any(tt.isinf_(cho)), tt.any(tt.isnan_(cho)))
        cond4 = tt.or_(tt.any(tt.isinf_(lcho)), tt.any(tt.isnan_(lcho)))
        return ifelse(cond1, np.float32(-1e30),
                      ifelse(cond2, np.float32(-1e30),
                             ifelse(cond3, np.float32(-1e30),
                                    ifelse(cond4, np.float32(-1e30), r))))

    def logp(self, value):
        return self.logp_cho(value, self.mu, self.cho, self.mapping)

    @property
    def cho(self):
        try:
            return cholesky_robust(self.cov) #tt_to_num
        except:
            raise sp.linalg.LinAlgError("not cholesky")
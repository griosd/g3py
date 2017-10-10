import numpy as np
import scipy as sp
import pymc3 as pm
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl
from theano.ifelse import ifelse
from .elliptical import EllipticalProcess, debug_p
from ..libs.tensors import tt_eval, cholesky_robust, debug
from .hypers.mappings import Identity
from .hypers import Freedom
from scipy import stats


class StudentTProcess(EllipticalProcess):

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'TP'
        if 'degree' not in kwargs:
            kwargs['degree'] = Freedom()
        super().__init__(*args, **kwargs)

    def th_define_process(self):
        super().th_define_process()
        self.distribution = WarpedStudentTDistribution(self.name,
                                                       mu=self.prior_location_inputs,
                                                       cov=self.prior_kernel_inputs,
                                                       freedom=self.th_freedom(prior=True),
                                                       mapping=self.f_mapping,
                                                       observed=self.th_outputs,
                                                       testval=self.th_outputs,
                                                       dtype=th.config.floatX)

    def th_scaling(self, prior=False, noise=False):
        if prior:
            return np.float32(1.0)
        np2 = np.float32(2.0)
        alpha = tsl.solve_lower_triangular(cholesky_robust(self.prior_kernel_inputs), self.mapping_outputs - self.prior_location_inputs)
        beta = alpha.T.dot(alpha)
        coeff = (self.th_freedom(prior=True) + beta - np2) / (self.th_freedom(prior=False) - np2)
        return coeff

    def th_variance(self, prior=False, noise=False):
        return super().th_variance(prior=prior, noise=noise) * self.th_scaling(prior=prior, noise=noise)

    def th_covariance(self, prior=False, noise=False):
        return super().th_covariance(prior=prior, noise=noise) * self.th_scaling(prior=prior, noise=noise)

    def quantiler(self, params=None, space=None, inputs=None, outputs=None, q=0.975, prior=False, noise=False):
        debug_p('quantiler')
        p = stats.t.ppf(q, df=self.freedom(params=params, space=space, inputs=inputs, outputs=outputs, prior=prior, noise=noise))
        gp_quantiler = self.location(params, space, inputs, outputs, prior=prior, noise=noise) + p*self.kernel_sd(params, space, inputs, outputs, prior=prior, noise=noise)
        return self.mapping(params, space, inputs, outputs=gp_quantiler) #self.f_mapping

    def sampler(self, params=None, space=None, inputs=None, outputs=None, samples=1, prior=False, noise=False):
        debug_p('sampler' + str(samples) + str(prior) + str(noise)+str(len(self.space)))
        if space is None:
            space = self.space
        free = self.freedom(params=params, space=space, inputs=inputs, outputs=outputs, prior=prior, noise=noise)
        rand = np.random.randn(len(space), samples) * stats.invgamma.rvs(a=free/2,
                                                                         scale=(free-2)/2,
                                                                         size=samples)
        qp_samples = self.location(params, space, inputs, outputs, prior=prior, noise=noise)[:, None] + \
                     self.cholesky(params, space, inputs, outputs, prior=prior, noise=noise).dot(rand)
        return np.array([self.mapping(params, space, inputs, outputs=k.T) for k in qp_samples.T]).T


class WarpedStudentTProcess(StudentTProcess):

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'WTP'
        if 'degree' not in kwargs:
            kwargs['degree'] = Freedom()
        super().__init__(*args, **kwargs)

    #TODO: fix mean
    def th_mean(self, prior=False, noise=False, simulations=None, n=10):
        debug_p('mean')
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX), borrow=False).dimshuffle([0, 'x'])
        w = th.shared(_w.astype(th.config.floatX), borrow=False)
        return self.gauss_hermite(lambda v: self.f_mapping(v), self.th_location(prior=prior, noise=noise),
                                  self.th_kernel_sd(prior=prior, noise=noise), a, w)

    #TODO: fix variance
    def th_variance(self, prior=False, noise=False, simulations=None, n=10):
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


class WarpedStudentTDistribution(pm.Continuous):
    def __init__(self, mu, cov, freedom, mapping, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov
        self.freedom = freedom
        self.mapping = mapping

    @classmethod
    def logp_cho(cls, value, mu, cho, freedom, mapping):
        delta = mapping.inv(value) - mu

        lcho = tsl.solve_lower_triangular(cho, delta)
        beta = lcho.T.dot(lcho)
        n = cho.shape[0].astype(th.config.floatX)

        np5 = np.float32(0.5)
        np2 = np.float32(2.0)
        npi = np.float32(np.pi)

        r1 = - np5 * (freedom + n) * tt.log1p(beta / (freedom-np2))
        r2 = ifelse(tt.le(np.float32(1e6), freedom), - n * np5 * np.log(np2 * npi),
                    tt.gammaln((freedom + n) * np5) - tt.gammaln(freedom * np5) - np5 * n * tt.log((freedom-np2) * npi))
        r3 = - tt.sum(tt.log(tnl.diag(cho)))
        det_m = mapping.logdet_dinv(value)

        r1 = debug(r1, name='r1', force=True)
        r2 = debug(r2, name='r2', force=True)
        r3 = debug(r3, name='r3', force=True)
        det_m = debug(det_m, name='det_m', force=True)

        r = r1 + r2 + r3 + det_m

        cond1 = tt.or_(tt.any(tt.isinf_(delta)), tt.any(tt.isnan_(delta)))
        cond2 = tt.or_(tt.any(tt.isinf_(det_m)), tt.any(tt.isnan_(det_m)))
        cond3 = tt.or_(tt.any(tt.isinf_(cho)), tt.any(tt.isnan_(cho)))
        cond4 = tt.or_(tt.any(tt.isinf_(lcho)), tt.any(tt.isnan_(lcho)))
        return ifelse(cond1, np.float32(-1e30),
                      ifelse(cond2, np.float32(-1e30),
                             ifelse(cond3, np.float32(-1e30),
                                    ifelse(cond4, np.float32(-1e30), r))))

    def logp(self, value):
        return self.logp_cho(value, self.mu, self.cho, self.freedom, self.mapping)
        #return tt_to_num(debug(self.logp_cho(value, self.mu, self.cho, self.freedom, self.mapping), 'logp_cho'), -np.inf, -np.inf)

    @property
    def cho(self):
        try:
            return cholesky_robust(self.cov)
        except:
            raise sp.linalg.LinAlgError("not cholesky")
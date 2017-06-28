import numpy as np
import scipy as sp
import pymc3 as pm
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl
from theano.ifelse import ifelse
from .stochastic import EllipticalProcess
from .gaussian import GaussianProcess
from ..libs.tensors import tt_to_num, cholesky_robust, debug
from .hypers.mappings import Identity
from scipy import stats


class StudentTProcess(GaussianProcess):

    def _define_process(self):
        super()._define_process()
        self.distribution = TransformedStudentTDistribution(self.name, mu=self.prior_location_inputs,
                                                            cov=self.prior_kernel_inputs, mapping=Identity(),
                                                            observed=self.th_outputs, testval=self.th_outputs,
                                                            dtype=th.config.floatX)

    def freedom(self, prior=False):
        if prior:
            return self.degree
        else:
            return self.degree + self.th_inputs.shape[1]

    def variance(self, prior=False, noise=False):
        beta = (self.mapping_outputs - self.location_inputs).T.dot(
            tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location))
        coeff = (self.freedom(prior=prior) + beta - 2) / (self.freedom(prior=prior) - 2)
        return super().variance(prior=prior, noise=noise) * coeff

    def quantiler(self, q=0.975, prior=False, noise=False):
        debug('quantiler')
        p = stats.t.ppf(q, df=self.freedom(prior=prior))
        return self.mapping(self.mean(prior=prior, noise=noise) + p * self.std(prior=prior, noise=noise))

    def sampler(self, nsamples=1, prior=False, noise=False):
        debug('sampler')
        rand = np.random.randn(len(self.th_space), nsamples) * stats.invgamma.rvs(df=self.freedom(prior=prior)/2,
                                                                                  loc=(self.freedom(prior=prior)-1)/2,
                                                                                  size=nsamples)
        return self.mapping(self.mean(prior=prior, noise=noise) + self.cholesky(prior=prior, noise=noise).dot(rand))


class TransformedStudentTDistribution(pm.Continuous):
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
        degree = freedom()

        det_m = mapping.logdet_dinv(value)

        r = -np.float32(0.5) * ((degree + n) * tt.log1p(beta / (degree-2)) + n * tt.log(np.float32((degree-2) * np.pi))) \
            - tt.sum(tt.log(tnl.diag(cho))) - tt.log(tt.gamma(degree / 2)) + tt.log(tt.gamma((degree + n) / 2)) + det_m

        cond1 = tt.or_(tt.any(tt.isinf_(delta)), tt.any(tt.isnan_(delta)))
        cond2 = tt.or_(tt.any(tt.isinf_(det_m)), tt.any(tt.isnan_(det_m)))
        cond3 = tt.or_(tt.any(tt.isinf_(lcho)), tt.any(tt.isnan_(lcho)))
        return ifelse(cond1, np.float32(-1e30), ifelse(cond2, np.float32(-1e30), ifelse(cond3, np.float32(-1e30), r)))

    def logp(self, value):
        return tt_to_num(debug(self.logp_cho(value, self.mu, self.cho, self.freedom, self.mapping), 'logp_cho'), -np.inf, -np.inf)

    @property
    def cho(self):
        try:
            return cholesky_robust(self.cov)
        except:
            raise sp.linalg.LinAlgError("not cholesky")
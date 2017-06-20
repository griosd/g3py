from .stochastic import *
from ..functions import Freedom, Kernel, Mean, Mapping, Identity
from ..libs import cholesky_robust, debug, ifelse
from pymc3.distributions.distribution import generate_samples
from scipy.stats._multivariate import multivariate_normal
from scipy.stats import t
from pymc3.distributions.special import gammaln


class StudentTProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, noise=True,
                 name='TP', inputs=None, outputs=None, hidden=None, file=None, precompile=False, *args, **kwargs):
        super().__init__(space=space, location=location, kernel=kernel, mapping=Identity(), noise=noise,
                         freedom=Freedom(bound=2), name=name, inputs=inputs, outputs=outputs, hidden=hidden, file=file, precompile=precompile, *args, **kwargs)

        self.prior_freedom = None
        self.posterior_freedom = None

    def _define_distribution(self):
        self.distribution = TTPDist(self.name, mu=self.location(self.inputs), cov=tt_to_cov(self.kernel.cov(self.inputs)),
                                    mapping=Identity(), freedom=self.freedom(), ttp=self, observed=self.outputs,
                                    testval=self.outputs, dtype=th.config.floatX)

    def _define_process(self):
        # Prior
        self.prior_freedom = self.freedom()
        self.prior_mean = self.location_space
        self.prior_covariance = self.kernel_f_space
        self.prior_cholesky = cholesky_robust(self.prior_covariance)
        self.prior_logp = TTPDist.logp_cho(self.random_th, self.prior_mean, self.prior_cholesky, self.mapping, self.prior_freedom)
        self.prior_distribution = tt.exp(self.prior_logp.sum())
        self.prior_variance = tnl.extract_diag(self.prior_covariance)
        self.prior_std = tt.sqrt(self.prior_variance)
        self.prior_noise = tt.sqrt(tnl.extract_diag(self.kernel_space))
        self.prior_logpred = TTPDist.logp_cho(self.random_th, self.prior_mean, nL.alloc_diag(self.prior_noise),
                                              self.mapping, self.prior_freedom)
        #Revisar
        sigma = self.random_scalar#np.float32(t.interval(0.95, 3.0)[1])
        self.prior_median = self.prior_mean
        self.prior_quantile_up = self.prior_mean + sigma * self.prior_std
        self.prior_quantile_down = self.prior_mean - sigma * self.prior_std
        self.prior_noise_up = self.prior_mean + sigma * self.prior_noise
        self.prior_noise_down = self.prior_mean - sigma * self.prior_noise
        self.prior_sampler = self.prior_mean + tt.sqrt(self.random_scalar) * self.prior_cholesky.dot(self.random_th)

        # Posterior
        self.posterior_freedom = self.prior_freedom + self.inputs.shape[0].astype(th.config.floatX)
        beta = (self.mapping_outputs - self.location_inputs).T.dot(tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        coeff = (self.prior_freedom + beta - 2)/(self.posterior_freedom - 2)
        self.posterior_mean = self.location_space + self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        self.posterior_covariance = coeff * (self.kernel_f.cov(self.space_th) - self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T)))
        self.posterior_cholesky = cholesky_robust(self.posterior_covariance)
        self.posterior_logp = TTPDist.logp_cho(self.random_th, self.posterior_mean, self.posterior_cholesky,
                                               self.mapping, self.posterior_freedom)
        self.posterior_distribution = tt.exp(self.posterior_logp.sum())
        self.posterior_variance = tnl.extract_diag(self.posterior_covariance)
        self.posterior_std = tt.sqrt(self.posterior_variance)
        self.posterior_noise = tt.sqrt(coeff*(tnl.extract_diag(self.kernel.cov(self.space_th) - self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T)))))
        self.posterior_logpred = TTPDist.logp_cho(self.random_th, self.posterior_mean,
                                                  nL.alloc_diag(self.posterior_noise), self.mapping, self.posterior_freedom)

        self.posterior_quantile_up = self.posterior_mean + sigma * self.posterior_std
        self.posterior_quantile_down = self.posterior_mean - sigma * self.posterior_std
        self.posterior_noise_up = self.posterior_mean + sigma * self.posterior_noise
        self.posterior_noise_down = self.posterior_mean - sigma * self.posterior_noise
        self.posterior_sampler = self.posterior_mean + tt.sqrt(self.random_scalar) * self.posterior_cholesky.dot(self.random_th)

    def subprocess(self, subkernel):
        k_cross = subkernel.cov(self.space_th, self.inputs_th)
        subprocess_mean = self.location_space + k_cross.dot(tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        params = [self.space_th, self.inputs_th, self.outputs_th] + self.model.vars
        return makefn(params, subprocess_mean, True)


class TTPDist(pm.Continuous):
    def __init__(self, mu, cov, mapping, freedom, ttp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov
        self.mapping = mapping
        self.freedom = freedom
        self.ttp = ttp


    @classmethod
    def logp_cho(cls, value, mu, cho, mapping, degree):
        delta = mapping.inv(value) - mu
        cond = tt.or_(tt.any(tt.isinf_(delta)), tt.any(tt.isnan_(delta)))

        _L = sL.solve_lower_triangular(cho, delta)
        beta = _L.T.dot(_L)
        n = cho.shape[0].astype(th.config.floatX)

        gmln1 = gammaln((degree + n) / 2.0) - gammaln(degree / 2.0) - np.float32(0.5) * n * tt.log((degree-2.0) * np.pi)
        gmln2 = approx_gammaln((degree + n) / 2.0) - approx_gammaln(degree / 2.0) - np.float32(0.5) * n * tt.log((degree-2.0) * np.pi)
        gmln = ifelse((degree + n) < 1e6, gmln1, gmln2)
        #gmln = th.printing.Print('gmln')(gmln)
        dot2 = np.float32(-0.5) * (degree + n) * tt.log1p(beta / (degree-2.0))
        det_k = - tt.sum(tt.log(nL.diag(cho)))
        det_m = mapping.logdet_dinv(value)
        r = gmln + dot2 + det_k + det_m
        cond1 = tt.or_(tt.isinf_(degree), tt.isnan_(degree))
        cond2 = tt.or_(tt.any(tt.isinf_(det_m)), tt.any(tt.isnan_(det_m)))
        cond3 = tt.or_(tt.any(tt.isinf_(_L)), tt.any(tt.isnan_(_L)))
        return ifelse(cond, np.float32(-1e30), ifelse(cond1, np.float32(-1e30), ifelse(cond2, np.float32(-1e30), ifelse(cond3, np.float32(-1e30), r))))

    def logp(self, value):
        return tt_to_num(debug(self.logp_cho(value, self.mu, self.cho, self.mapping, self.freedom), 'logp_cho'), -np.inf, -np.inf)

    @property
    def cho(self):
        try:
            return cholesky_robust(self.cov)
        except:
            raise sp.linalg.LinAlgError("not cholesky")


def approx_gammaln(z):
    return (z-np.float32(0.5))*tt.log(z) - z #+0.5*np.log(2*np.pi)
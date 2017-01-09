from .stochastic import *
from ..functions import Freedom, Kernel, Mean, Mapping, Identity
from ..libs import cholesky_robust
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl


class StudentTProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, noise=True,
                 name=None, inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=Identity(), noise=noise,
                         freedom=Freedom(bound=2), name=name, inputs=inputs, outputs=outputs, hidden=hidden)

        self.prior_freedom = None
        self.posterior_freedom = None

    def define_distribution(self):
        self.distribution = STPDist(self.name, mu=self.location(self.inputs), cov=tt_to_cov(self.kernel.cov(self.inputs)), mapping = Identity(),
                       freedom=self.freedom, stp=self, observed=self.outputs, testval=self.outputs, dtype=th.config.floatX)

    def define_process(self):
        # Prior
        self.prior_freedom = self.freedom()
        self.prior_mean = self.location_space
        self.prior_covariance = self.kernel_f_space * self.prior_freedom
        self.prior_variance = tnl.extract_diag(self.prior_covariance)
        self.prior_std = tt.sqrt(self.prior_variance)
        self.prior_noise = tt.sqrt(tnl.extract_diag(self.kernel_space * self.prior_freedom))
        self.prior_median = self.prior_mean

        sigma = 2
        self.prior_quantile_up = self.prior_mean + sigma * self.prior_std
        self.prior_quantile_down = self.prior_mean - sigma * self.prior_std
        self.prior_noise_up = self.prior_mean + sigma * self.prior_noise
        self.prior_noise_down = self.prior_mean - sigma * self.prior_noise

        self.prior_sampler = self.prior_mean + self.random_scalar * cholesky_robust(self.prior_covariance).dot(self.random_th)

        # Posterior
        self.posterior_freedom = self.prior_freedom + self.inputs.shape[1]
        beta = (self.mapping_outputs - self.location_inputs).T.dot(tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        coeff = (self.prior_freedom + beta - 2)/(self.posterior_freedom - 2)
        self.posterior_mean = self.location_space + self.kernel_f_space_inputs.dot( tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        self.posterior_covariance = coeff * (self.kernel_f.cov(self.space_th) - self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T)))
        self.posterior_variance = tnl.extract_diag(self.posterior_covariance)
        self.posterior_std = tt.sqrt(self.posterior_variance)
        self.posterior_noise = coeff * tt.sqrt(tnl.extract_diag(self.kernel.cov(self.space_th) - self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))))
        self.posterior_median = self.posterior_mean
        self.posterior_quantile_up = self.posterior_mean + sigma * self.posterior_std
        self.posterior_quantile_down = self.posterior_mean - sigma * self.posterior_std
        self.posterior_noise_up = self.posterior_mean + sigma * self.posterior_noise
        self.posterior_noise_down = self.posterior_mean - sigma * self.posterior_noise
        self.posterior_sampler = self.posterior_mean + self.random_scalar * cholesky_robust(self.posterior_covariance).dot(self.random_th)




class TransformedStudentTProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, mapping: Mapping=None, noise=True,
                 freedom=None, name=None, inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=mapping, noise=noise,
                         freedom=freedom, name=name, inputs=inputs, outputs=outputs, hidden=hidden)



class STPDist(pm.Continuous):
    def __init__(self, mu, cov, mapping, freedom, stp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov
        self.mapping = mapping
        self.freedom = freedom
        self.stp = stp

    @classmethod
    def logp_cov(cls, value, mu, cov, freedom):  # es más rápido pero se cae
        delta = value - mu
        beta = delta.T.dot(sL.solve(cov, delta))
        n = cov.shape[0].astype(th.config.floatX)
        degree = freedom()
        return -np.float32(0.5) * (tt.log(nL.det(cov)) + (degree + n) * tt.log1p( beta / (degree-2))
                                   + n * tt.log(np.float32((degree-2) * np.pi))) \
               - tt.log(tt.gamma(degree / 2)) + tt.log(tt.gamma((degree + n) / 2))

    @classmethod
    def logp_cho(cls, value, mu, cho, freedom):
        delta = value - mu
        L = sL.solve_lower_triangular(cho, delta)
        beta = L.T.dot(L)
        n = cho.shape[0].astype(th.config.floatX)
        degree = freedom()
        return -np.float32(0.5) * ((degree + n) * tt.log1p(beta / (degree-2)) + n * tt.log(np.float32((degree-2) * np.pi))) \
               - tt.sum(tt.log(nL.diag(cho))) - tt.log(tt.gamma(degree / 2)) + tt.log(tt.gamma((degree + n) / 2))

    def logp(self, value):
        if False:
            return tt_to_num(debug(self.logp_cov(value, self.mu, self.cov, self.freedom), 'logp_cov'), -np.inf, -np.inf)
        else:
            return tt_to_num(debug(self.logp_cho(value, self.mu, self.cho, self.freedom), 'logp_cho'), -np.inf, -np.inf)

    @property
    def cho(self):
        try:
            return cholesky_robust(self.cov)
        except:
            raise sp.linalg.LinAlgError("not cholesky")


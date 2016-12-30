from .stochastic import *
from ..functions import Kernel, Mean, Mapping, Identity
from ..libs import cholesky_robust
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl


class GaussianProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, noise=True,
                 name='GP', inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=Identity(), noise=noise,
                         freedom=None, name=name, inputs=inputs, outputs=outputs, hidden=hidden)

    def define_process(self):
        # Prior
        self.prior_mean = self.location_space
        self.prior_covariance = self.kernel_f_space
        self.prior_variance = tnl.extract_diag(self.prior_covariance)
        self.prior_std = tt.sqrt(self.prior_variance)
        self.prior_noise = tt.sqrt(tnl.extract_diag(self.kernel_space))
        self.prior_median = self.prior_mean
        self.prior_quantile_up = self.prior_mean + 1.96 * self.prior_std
        self.prior_quantile_down = self.prior_mean - 1.96 * self.prior_std
        self.prior_noise_up = self.prior_mean + 1.96 * self.prior_noise
        self.prior_noise_down = self.prior_mean - 1.96 * self.prior_noise
        self.prior_sampler = self.prior_mean + cholesky_robust(self.prior_covariance).dot(self.random_th)

        # Posterior
        self.posterior_mean = self.location_space + self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        self.posterior_covariance = self.kernel_f.cov(self.space_th) - self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))
        self.posterior_variance = tnl.extract_diag(self.posterior_covariance)
        self.posterior_std = tt.sqrt(self.posterior_variance)
        self.posterior_noise = tt.sqrt(tnl.extract_diag(self.kernel.cov(self.space_th) - self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))))
        self.posterior_median = self.posterior_mean
        self.posterior_quantile_up = self.posterior_mean + 1.96 * self.posterior_std
        self.posterior_quantile_down = self.posterior_mean - 1.96 * self.posterior_std
        self.posterior_noise_up = self.posterior_mean + 1.96 * self.posterior_noise
        self.posterior_noise_down = self.posterior_mean - 1.96 * self.posterior_noise
        self.posterior_sampler = self.posterior_mean + cholesky_robust(self.posterior_covariance).dot(self.random_th)

        # TODO
        def marginal(self):
            value = tt.vector('marginal_gp')
            value.tag.test_value = zeros(1)
            delta = value - self.location(self.space)
            cov = self.kernel.cov(self.space)
            cho = cholesky_robust(cov)
            L = tsl.solve_lower_triangular(cho, delta)
            return value, tt.exp(-np.float32(0.5) * (cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                                     + L.T.dot(L)) - tt.sum(tt.log(tnl.extract_diag(cho))))
        # TODO
        def subprocess(self, subkernel, cov=False, noise=False):
            k_ni = subkernel.cov(self.space, self.inputs)
            self.subprocess_mean = self.mean(self.space) + k_ni.dot(tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
            self.subprocess_covariance = self.kernel_f.cov(self.space) - k_ni.dot(tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))
            self.subprocess_noise = self.kernel.cov(self.space) - k_ni.dot(tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))


class TransformedGaussianProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, mapping: Mapping=None, noise=True,
                 name='TGP', inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=mapping, noise=noise,
                         freedom=None, name=name, inputs=inputs, outputs=outputs, hidden=hidden)

        self.latent_prior_mean = None
        self.latent_prior_covariance = None
        self.latent_prior_std = None
        self.latent_prior_noise = None

        self.latent_posterior_mean = None
        self.latent_posterior_covariance = None
        self.latent_posterior_std = None
        self.latent_posterior_noise = None

    def define_process(self, n=20):
        # Gauss-Hermite
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX), borrow=True).dimshuffle([0, 'x'])
        w = th.shared(_w.astype(th.config.floatX), borrow=True)

        # Latent
        self.latent_prior_mean = self.location_space
        self.latent_prior_covariance = self.kernel_f_space
        self.latent_prior_std = np.sqrt(tnl.extract_diag(self.latent_prior_covariance))
        self.latent_prior_noise = np.sqrt(tnl.extract_diag(self.kernel_space))
        self.latent_posterior_mean = self.location_space + self.kernel_f_space_inputs.dot(tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        self.latent_posterior_covariance = self.kernel_f.cov(self.space_th) - self.kernel_f_space_inputs.dot(tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))
        self.latent_posterior_std = np.sqrt(tnl.extract_diag(self.latent_posterior_covariance))
        self.latent_posterior_noise = np.sqrt(tnl.extract_diag(self.kernel.cov(self.space_th) - self.kernel_f_space_inputs.dot(tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))))

        # Prior
        self.prior_mean = gauss_hermite(self.mapping, self.latent_prior_mean, self.latent_prior_std, a, w)
        self.prior_variance = gauss_hermite(lambda v: self.mapping(v) ** 2, self.latent_prior_mean,
                                            self.latent_prior_std, a, w) - self.prior_mean ** 2
        self.prior_std = tt.sqrt(self.prior_variance)
        self.prior_covariance = None
        self.prior_noise = None
        self.prior_median = self.mapping(self.latent_prior_mean)
        self.prior_quantile_up = self.mapping(self.latent_prior_mean + 1.96 * self.latent_prior_std)
        self.prior_quantile_down = self.mapping(self.latent_prior_mean - 1.96 * self.latent_prior_std)
        self.prior_noise_up = self.mapping(self.latent_prior_mean + 1.96 * self.latent_prior_noise)
        self.prior_noise_down = self.mapping(self.latent_prior_mean - 1.96 * self.latent_prior_noise)
        self.prior_sampler = self.mapping(self.latent_prior_mean + cholesky_robust(self.latent_prior_covariance).dot(self.random_th))

        # Posterior
        self.posterior_mean = gauss_hermite(self.mapping, self.latent_posterior_mean, self.latent_posterior_std, a, w)
        self.posterior_variance = gauss_hermite(lambda v: self.mapping(v) ** 2, self.latent_posterior_mean,
                                                self.latent_posterior_std, a, w) - self.posterior_mean ** 2
        self.posterior_std = tt.sqrt(self.posterior_variance)
        self.posterior_covariance = None
        self.posterior_noise = None
        self.posterior_median = self.mapping(self.latent_posterior_mean)
        self.posterior_quantile_up = self.mapping(self.latent_posterior_mean + 1.96 * self.latent_posterior_std)
        self.posterior_quantile_down = self.mapping(self.latent_posterior_mean - 1.96 * self.latent_posterior_std)
        self.posterior_noise_up = self.mapping(self.latent_posterior_mean + 1.96 * self.latent_posterior_noise)
        self.posterior_noise_down = self.mapping(self.latent_posterior_mean - 1.96 * self.latent_posterior_noise)
        self.posterior_sampler = self.mapping(self.latent_posterior_mean + cholesky_robust(self.latent_posterior_covariance).dot(self.random_th))

        # TODO

        # TODO
        def marginal_tgp(self):
            value = tt.vector('marginal_tgp')
            value.tag.test_value = zeros(1)
            delta = self.mapping.inv(value) - self.mean(self.space)
            cov = self.kernel.cov(self.space)
            cho = cholesky_robust(cov)
            L = tsl.solve_lower_triangular(cho, delta)
            return value, tt.exp(-np.float32(0.5) * (cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                                     + L.T.dot(L)) - tt.sum(tt.log(tnl.extract_diag(cho))) + self.mapping.logdet_dinv(value))
        # TODO
        def subprocess(self, subkernel, cov=False, noise=False):
            k_ni = subkernel.cov(self.space, self.inputs)
            self.subprocess_mean = self.mean(self.space) + k_ni.dot(tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
            self.subprocess_covariance = self.kernel_f.cov(self.space) - k_ni.dot(tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))
            self.subprocess_noise = self.kernel.cov(self.space) - k_ni.dot(tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))


def gauss_hermite(f, mu, sigma, a, w):
    return tt.dot(w, f(mu + sigma * np.sqrt(2) * a)) / np.sqrt(np.pi)
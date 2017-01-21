from .stochastic import *
from ..functions import Kernel, Mean, Mapping, Identity
from ..libs import cholesky_robust, debug
from pymc3.distributions.distribution import generate_samples
from scipy.stats._multivariate import multivariate_normal


class GaussianProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, noise=True,
                 name='GP', inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=Identity(), noise=noise,
                         freedom=None, name=name, inputs=inputs, outputs=outputs, hidden=hidden)

    def define_distribution(self):
        self.distribution = TGPDist(self.name, mu=self.location(self.inputs), cov=tt_to_cov(self.kernel.cov(self.inputs)),
                            mapping=Identity(), tgp=self, observed=self.outputs, testval=self.outputs, dtype=th.config.floatX)

    def define_process(self):
        # Prior
        self.prior_mean = self.location_space
        self.prior_covariance = self.kernel_f_space
        self.prior_cholesky = cholesky_robust(self.prior_covariance)
        self.prior_logp = TGPDist.logp_cho(self.random_th, self.prior_mean, self.prior_cholesky, self.mapping)
        self.prior_distribution = tt.exp(self.prior_logp.sum())
        self.prior_variance = tnl.extract_diag(self.prior_covariance)
        self.prior_std = tt.sqrt(self.prior_variance)
        self.prior_noise = tt.sqrt(tnl.extract_diag(self.kernel_space))
        self.prior_logpred = TGPDist.logp_cho(self.random_th, self.prior_mean, nL.alloc_diag(self.prior_noise),
                                              self.mapping)

        self.prior_median = self.prior_mean
        self.prior_quantile_up = self.prior_mean + 1.96 * self.prior_std
        self.prior_quantile_down = self.prior_mean - 1.96 * self.prior_std
        self.prior_noise_up = self.prior_mean + 1.96 * self.prior_noise
        self.prior_noise_down = self.prior_mean - 1.96 * self.prior_noise
        self.prior_sampler = self.prior_mean + self.prior_cholesky.dot(self.random_th)

        # Posterior
        self.posterior_mean = self.location_space + self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        self.posterior_covariance = self.kernel_f.cov(self.space_th) - self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))
        self.posterior_cholesky = cholesky_robust(self.posterior_covariance)
        self.posterior_logp = TGPDist.logp_cho(self.random_th, self.posterior_mean, self.posterior_cholesky, self.mapping)
        self.posterior_distribution = tt.exp(self.posterior_logp.sum())
        self.posterior_variance = tnl.extract_diag(self.posterior_covariance)
        self.posterior_std = tt.sqrt(self.posterior_variance)
        self.posterior_noise = tt.sqrt(tnl.extract_diag(self.kernel.cov(self.space_th) - self.kernel_f_space_inputs.dot(
            tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))))
        self.posterior_logpred = TGPDist.logp_cho(self.random_th, self.posterior_mean, nL.alloc_diag(self.posterior_noise), self.mapping)
        self.posterior_median = self.posterior_mean
        self.posterior_quantile_up = self.posterior_mean + 1.96 * self.posterior_std
        self.posterior_quantile_down = self.posterior_mean - 1.96 * self.posterior_std
        self.posterior_noise_up = self.posterior_mean + 1.96 * self.posterior_noise
        self.posterior_noise_down = self.posterior_mean - 1.96 * self.posterior_noise
        self.posterior_sampler = self.posterior_mean + self.posterior_cholesky.dot(self.random_th)

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

    def define_distribution(self):
        self.distribution = TGPDist(self.name, mu=self.location(self.inputs), cov=tt_to_cov(self.kernel.cov(self.inputs)),
                                    mapping=self.mapping, tgp=self, observed=self.outputs, testval=self.outputs, dtype=th.config.floatX)

    def define_process(self, n=10):
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

        self.prior_cholesky = cholesky_robust(self.latent_prior_covariance)
        self.prior_logp = TGPDist.logp_cho(self.random_th, self.latent_prior_mean, self.prior_cholesky, self.mapping)
        self.prior_logpred = TGPDist.logp_cho(self.random_th, self.latent_prior_mean, nL.alloc_diag(self.latent_prior_noise), self.mapping)
        self.prior_distribution = tt.exp(self.prior_logp.sum())

        self.posterior_cholesky = cholesky_robust(self.latent_posterior_covariance)
        self.posterior_logp = TGPDist.logp_cho(self.random_th, self.latent_posterior_mean, self.posterior_cholesky, self.mapping)
        self.posterior_logpred = TGPDist.logp_cho(self.random_th, self.latent_posterior_mean, nL.alloc_diag(self.latent_posterior_noise), self.mapping)
        self.posterior_distribution = tt.exp(self.posterior_logp.sum())
        print('Latent OK')

        # Prior
        self.prior_mean = gauss_hermite(self.mapping, self.latent_prior_mean, self.latent_prior_std, a, w)
        self.prior_variance = gauss_hermite(lambda v: self.mapping(v) ** 2, self.latent_prior_mean,
                                            self.latent_prior_std, a, w) - self.prior_mean ** 2
        self.prior_std = tt.sqrt(self.prior_variance)
        self.prior_covariance = None
        self.prior_noise = tt.sqrt(gauss_hermite(lambda v: self.mapping(v) ** 2, self.latent_prior_mean,
                                                self.latent_prior_noise, a, w) - self.prior_mean ** 2)
        self.prior_median = self.mapping(self.latent_prior_mean)
        self.prior_quantile_up = self.mapping(self.latent_prior_mean + 1.96 * self.latent_prior_std)
        self.prior_quantile_down = self.mapping(self.latent_prior_mean - 1.96 * self.latent_prior_std)
        self.prior_noise_up = self.mapping(self.latent_prior_mean + 1.96 * self.latent_prior_noise)
        self.prior_noise_down = self.mapping(self.latent_prior_mean - 1.96 * self.latent_prior_noise)
        self.prior_sampler = self.mapping(self.latent_prior_mean + cholesky_robust(self.latent_prior_covariance).dot(self.random_th))
        print('Prior OK')


        # Posterior
        self.posterior_mean = gauss_hermite(self.mapping, self.latent_posterior_mean, self.latent_posterior_std, a, w)
        self.posterior_variance = gauss_hermite(lambda v: self.mapping(v) ** 2, self.latent_posterior_mean,
                                                self.latent_posterior_std, a, w) - self.posterior_mean ** 2
        self.posterior_std = tt.sqrt(self.posterior_variance)
        self.posterior_covariance = None
        self.posterior_noise = tt.sqrt(gauss_hermite(lambda v: self.mapping(v) ** 2, self.latent_posterior_mean,
                                                self.latent_posterior_noise, a, w) - self.posterior_mean ** 2)
        self.posterior_median = self.mapping(self.latent_posterior_mean)
        self.posterior_quantile_up = self.mapping(self.latent_posterior_mean + 1.96 * self.latent_posterior_std)
        self.posterior_quantile_down = self.mapping(self.latent_posterior_mean - 1.96 * self.latent_posterior_std)
        self.posterior_noise_up = self.mapping(self.latent_posterior_mean + 1.96 * self.latent_posterior_noise)
        self.posterior_noise_down = self.mapping(self.latent_posterior_mean - 1.96 * self.latent_posterior_noise)
        self.posterior_sampler = self.mapping(self.latent_posterior_mean + cholesky_robust(self.latent_posterior_covariance).dot(self.random_th))
        print('Posterior OK')

        # TODO
        def subprocess(self, subkernel, cov=False, noise=False):
            k_ni = subkernel.cov(self.space, self.inputs)
            self.subprocess_mean = self.mean(self.space) + k_ni.dot(tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
            self.subprocess_covariance = self.kernel_f.cov(self.space) - k_ni.dot(tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))
            self.subprocess_noise = self.kernel.cov(self.space) - k_ni.dot(tsl.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))


def gauss_hermite(f, mu, sigma, a, w):
    grille = mu + sigma * np.sqrt(2).astype(th.config.floatX) * a
    return tt.dot(w, f(grille.flatten()).reshape(grille.shape)) / np.sqrt(np.pi).astype(th.config.floatX)


class TGPDist(pm.Continuous):
    def __init__(self, mu, cov, mapping, tgp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov
        self.mapping = mapping
        self.tgp = tgp

    @classmethod
    def logp_cov(cls, value, mu, cov, mapping):  # es más rápido pero se cae
        delta = tt_to_num(mapping.inv(value)) - mu
        return -np.float32(0.5) * (tt.log(nL.det(cov)) + delta.T.dot(sL.solve(cov, delta))
                                   + cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))) \
               + mapping.logdet_dinv(value)

    @classmethod
    def logp_cho(cls, value, mu, cho, mapping):
        delta = tt_to_num(mapping.inv(value) - mu)
        L = sL.solve_lower_triangular(cho, delta)
        return -np.float32(0.5) * (cho.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                   + L.T.dot(L)) - tt.sum(tt.log(nL.diag(cho))) + mapping.logdet_dinv(value)

    def logp(self, value):
        if False:
            return tt_to_num(debug(self.logp_cov(value, self.mu, self.cov, self.mapping), 'logp_cov'), -np.inf, -np.inf)
        else:
            return tt_to_num(debug(self.logp_cho(value, self.mu, self.cho, self.mapping), 'logp_cho'), -np.inf, -np.inf)

    @property
    def cho(self):
        try:
            return tt_to_num(cholesky_robust(self.cov))
        except:
            raise sp.linalg.LinAlgError("not cholesky")

    def random(self, point=None, size=None):
        point = self.tgp.get_params(self.tgp, point)
        mu = self.tgp.compiles['mean'](**point)
        cov = self.tgp.compiles['covariance'](**point)

        def _random(mean, cov, size=None):
            return self.compiles['trans'](multivariate_normal.rvs(mean, cov, None if size == mean.shape else size), **point)
        samples = generate_samples(_random, mean=mu, cov=cov, dist_shape=self.shape, broadcast_shape=mu.shape, size=size)
        return samples


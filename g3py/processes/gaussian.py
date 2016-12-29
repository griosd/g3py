from .stochastic import *
from g3py import StochasticProcess, Kernel, Mean, Mapping, TGPDist


class GaussianProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, noise=True,
                 name=None, inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=Identity(), noise=noise,
                         freedom=None, name=name, inputs=inputs, outputs=outputs, hidden=hidden)

    def define_process(self):
        # super().define_process()

        # Prior
        self.prior_mean = self.location_space
        self.prior_covariance = self.kernel_f_space
        self.prior_variance = nL.extract_diag(self.prior_covariance)
        self.prior_noise = nL.extract_diag(self.kernel_space)
        self.prior_quantile_up = self.prior_mean + 1.96*tt.sqrt(self.prior_covariance)
        self.prior_quantile_down = self.prior_mean - 1.96*tt.sqrt(self.prior_covariance)
        self.prior_noise_up = self.prior_mean + 1.96*tt.sqrt(self.prior_noise)
        self.prior_noise_down = self.prior_mean - 1.96*tt.sqrt(self.prior_noise)

        # Posterior
        self.posterior_mean = self.location_space + self.kernel_f_space_inputs.dot(sL.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        self.posterior_covariance = self.kernel_f.cov(self.space) - self.kernel_f_space_inputs.dot(sL.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))
        self.posterior_variance = nL.extract_diag(self.prior_covariance)
        self.posterior_noise = nL.extract_diag(self.kernel.cov(self.space) - self.kernel_f_space_inputs.dot(sL.solve(self.kernel_inputs, self.kernel_f_space_inputs.T)))
        self.posterior_quantile_up = self.posterior_mean + 1.96*tt.sqrt(self.prior_covariance)
        self.posterior_quantile_down = self.posterior_mean - 1.96*tt.sqrt(self.prior_covariance)
        self.posterior_noise_up = self.posterior_mean + 1.96*tt.sqrt(self.prior_noise)
        self.posterior_noise_down = self.posterior_mean - 1.96*tt.sqrt(self.prior_noise)

        self.distribution = TGPDist(self.name, mu=self.location_inputs, cov=self.kernel_inputs,
                                    mapping=self.mapping, tgp=self, observed=self.outputs,
                                    testval=self.outputs, dtype=th.config.floatX)

    def marginal(self):
        value = tt.vector('marginal_gp')
        value.tag.test_value = zeros(1)
        delta = value - self.location(self.space)
        cov = self.kernel.cov(self.space)
        cho = cholesky_robust(cov)
        L = sL.solve_lower_triangular(cho, delta)
        return value, tt.exp(-np.float32(0.5) * (cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                                 + L.T.dot(L)) - tt.sum(tt.log(nL.extract_diag(cho))))

    def prior(self):
        pass





    def posterior(self, cov=False, noise=False):
        pass


    def subprocess(self, subkernel, cov=False, noise=False):
        k_ni = subkernel.cov(self.space, self.inputs)
        self.subprocess_mean = self.mean(self.space) + k_ni.dot(sL.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        self.subprocess_covariance = self.kernel_f.cov(self.space) - k_ni.dot(sL.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))
        self.subprocess_noise = self.kernel.cov(self.space) - k_ni.dot(sL.solve(self.kernel_inputs, self.kernel_f_space_inputs.T))

    def sampler(self):
        random = tt.vector('random')
        random.tag.test_value = zeros(self.space_x.shape[0])
        if self.outputs is None:
            mu, var, cov = self.prior_gp(cov=True)
        else:
            mu, var, cov = self.posterior_gp(cov=True)
        cho = cholesky_robust(cov)
        return random, mu + cho.dot(random)

    def sample(self, params, samples=1):
        S = np.empty((len(self.space_x), samples))
        for i in range(samples):
            S[:, i] = self.compiles['sampler_gp'](np.random.randn(len(self.space_x)), **params)
        return S

    def predict(self, params, mean=True, var=True, cov=False, noise=False):
        r = list()
        if mean:
            r.append(self.compiles['mean'](**params))
        if var:
            r.append(self.compiles['variance'](**params))
        if cov:
            r.append(self.compiles['covariance'](**params))
        if noise:
            r.append(self.compiles['noise'](**params))
        return r

    def plot_gp_moments(self, params, sigma=1.96):
        mean = self.compiles['mean'](**params)
        variance = self.compiles['variance'](**params)
        noise = self.compiles['noise'](**params)
        std = np.sqrt(variance)
        wn = np.sqrt(noise)

        plt.plot(self.space_t, mean, label='Mean')
        plt.fill_between(self.space_t, mean + sigma * std, mean - sigma * std, alpha=0.2, label='95%')
        plt.fill_between(self.space_t, mean + sigma * wn, mean - sigma * wn, alpha=0.1, label='noise')
        return mean, variance, noise

    def plot_gp_samples(self, params, samples=1):
        for i in range(samples):
            plt.plot(self.space_t, self.compiles['sampler_gp'](np.random.randn(len(self.space_x)), **params), alpha=0.6)

    def plot_gp(self, params, title=None, samples=0, big=True, scores=False, loc=1):
        #if self.space_t is not self.space_x:
        #    plt.figure(0)
        #    self.plot_space()
        #plt.figure(1)
        self.plot_data(big)
        self.plot_gp_moments(params)
        self.plot_gp_samples(params, samples)
        if scores:
            self.description['scores'] = self.scores(params)
            score_title = 'll_train={:.2f} | ll_test={:.2f}'.format(self.description['scores']['logp_train'],
                                                                self.description['scores']['logp_test'])
        else:
            score_title = ''
        if title is None:
            title = self.description['title']
        text_plot(title + ': ' + score_title,  self.description['x'],  self.description['y'], loc=loc)


class TransformedGaussianProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, mapping: Mapping=None, noise=True,
                 name=None, inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=mapping, noise=noise,
                         freedom=None, name=name, inputs=inputs, outputs=outputs, hidden=hidden)

    def marginal_tgp(self):
        value = tt.vector('marginal_tgp')
        value.tag.test_value = zeros(1)
        delta = self.mapping.inv(value) - self.mean(self.space)
        cov = self.kernel.cov(self.space)
        cho = cholesky_robust(cov)
        L = sL.solve_lower_triangular(cho, delta)
        return value, tt.exp(-np.float32(0.5) * (cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                                 + L.T.dot(L)) - tt.sum(tt.log(nL.extract_diag(cho))) + self.mapping.logdet_dinv(value))

    def prior_quantiles_tgp(self, sigma=1.96, noise=False):
        mu, var = self.prior_gp(cov=False, noise=noise)
        std = tt.sqrt(var)
        return self.mapping(mu), self.mapping(mu - sigma * std), self.mapping(mu + sigma * std)

    def prior_moments_tgp(self, n=20):
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX), borrow=True).dimshuffle([0, 'x'])
        w = th.shared(_w.astype(th.config.floatX), borrow=True)
        mu, var = self.prior_gp()
        std = tt.sqrt(var)
        return self.gauss_hermite(self.mapping, mu, std, a, w), self.gauss_hermite(lambda v: self.mapping(v) ** 2, mu,
                                                                                   std, a, w)

    def posterior_quantiles_tgp(self, sigma=1.96, noise=False):
        mu, var = self.posterior_gp(noise=noise)
        std = tt.sqrt(var)
        return self.mapping(mu), self.mapping(mu - sigma * std), self.mapping(mu + sigma * std)

    def posterior_moments_tgp(self, n=20):
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX), borrow=True).dimshuffle([0, 'x'])
        w = th.shared(_w.astype(th.config.floatX), borrow=True)
        mu, var = self.posterior_gp()
        std = tt.sqrt(var)
        return self.gauss_hermite(self.mapping, mu, std, a, w), self.gauss_hermite(lambda v: self.mapping(v) ** 2, mu,
                                                                                   std, a, w)

    def sampler_tgp(self):
        random, samples = self.sampler_gp()
        return random, self.mapping(samples)

    def plot_tgp_quantiles(self, params):
        med = self.compiles['median'](**params)
        Iup = self.compiles['I_up'](**params)
        Idown = self.compiles['I_down'](**params)
        noise_up = self.compiles['noise_up'](**params)
        noise_down = self.compiles['noise_down'](**params)
        plt.plot(self.space_t, med, label='Median')
        plt.fill_between(self.space_t, Iup, Idown, alpha=0.2, label='95%')
        plt.fill_between(self.space_t, noise_up, noise_down, alpha=0.1, label='Noise')
        return med, Iup, Idown

    def plot_tgp_moments(self, params, std=False):
        moment1 = self.compiles['m1'](**params)
        moment2 = self.compiles['m2'](**params)
        sigma = np.sqrt(moment2-moment1**2)
        plt.plot(self.space_t, moment1, label='Mean')
        if std:
            plt.plot(self.space_t, moment1 + 2.0 * sigma, '--k', alpha=0.2, label='4.0 std')
            plt.plot(self.space_t, moment1 - 2.0 * sigma, '--k', alpha=0.2)
        return moment1, moment2

    def plot_tgp_samples(self, params, samples=1):
        for i in range(samples):
            plt.plot(self.space_t, self.compiles['sampler_tgp'](np.random.randn(len(self.space_x)), **params), alpha=0.6)

    def plot_tgp(self, params, title=None, samples=0, std=False, big=True, scores=False, loc=1):
        #if self.space_t is not self.space_x:
        #    plt.figure(0)
        #    self.plot_space()
        #plt.figure(1)
        self.plot_data(big)
        self.plot_tgp_moments(params, std)
        self.plot_tgp_quantiles(params)
        self.plot_tgp_samples(params, samples)
        if scores: #TODO: Check Scores
            self.description['scores'] = self.scores(params)
            score_title = 'll_train={:.2f} | ll_test={:.2f} | mse={:.2f}  | bias={:.2f}'.format(self.description['scores']['logp_train'],
                                                                                          self.description['scores']['logp_test'],
                                                                                          self.description['scores']['mse_test'],
                                                                                          self.description['scores']['mab_test'])
            score_title = 'll_train={:.2f} | ll_test={:.2f}'.format(self.description['scores']['logp_train'], self.description['scores']['logp_test'])
        else:
            score_title = ''
        if title is None:
            title = self.description['title']
        text_plot(title + ': ' + score_title,  self.description['x'],  self.description['y'], loc=loc)


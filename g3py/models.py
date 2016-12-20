import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pymc3 as pm
import theano as th
import IPython.html.widgets as widgets
from .functions.hypers import zeros, trans_hypers
from .functions.means import Mean
from .functions.kernels import Kernel, KernelSum, WN
from .functions.mappings import Mapping, Identity
from .libs.tensors import cholesky_robust, makefn
from .libs.plots import text_plot
from theano import tensor as tt
from theano.sandbox import linalg as sT


class Model(pm.Model):

    @classmethod
    def get_contexts(cls):
        return pm.Model.get_contexts()

    @classmethod
    def get_context(cls):
        return pm.Model.get_context()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior = False

    def use_prior(self, prior=True):
        self.prior = prior

    def point_to_eval(self, point):
        r = dict()
        for k, v in point.items():
            r[self[k]] = v
        return r

    def point_to_In(self, point):
        r = list()
        for k, v in point.items():
            r.append(th.In(self[k], value=v))
        return r

    @property
    def logpt(self):
        if self.prior:
            return self.log_posterior_t
        else:
            return self.log_likelihood_t

    @property
    def log_posterior_t(self):
        """Theano scalar of log-posterior of the model"""
        factors = [var.logpt for var in self.basic_RVs] + self.potentials
        return tt.add(*map(tt.sum, factors))

    @property
    def log_likelihood_t(self):
        """Theano scalar of log-likelihood of the model"""
        factors = [var.logpt for var in self.observed_RVs]
        return tt.add(*map(tt.sum, factors))


class TGP:
    def __init__(self, space, mean: Mean, kernel: Kernel, mapping: Mapping, noise=True, name=None, hidden=None):
        self.model = Model.get_context()
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.space_x = space.astype(th.config.floatX)
        self.space = th.shared(self.space_x, self.name + '_Space')
        self.mean = mean
        if noise:
            self.kernel_f = kernel
            self.kernel = KernelSum(self.kernel_f, WN(name='Noise'))
        else:
            self.kernel_f = kernel
            self.kernel = self.kernel_f
        self.mapping = mapping

        self.inputs = None
        self.outputs = None
        self.mean_inputs = None
        self.cov_inputs = None
        self.inv_outputs = None
        self.distribution = None
        self.inputs_test = None
        self.outputs_test = None
        self.inv_outputs = None
        self._widget_params = None

        self.compiles = {}
        self.description = {}
        self.hidden = hidden
        self.check_hypers()

    def check_hypers(self):
        self.mean.check_hypers(self.name+'_')
        self.kernel.check_hypers(self.name+'_')
        self.mapping.check_hypers(self.name+'_')

    def set_space(self, space):
        self.space_x = space.astype(th.config.floatX)
        self.space.set_value(self.space_x)

    def describe(self, title, x, y, text=''):
        self.description['title'] = title
        self.description['x'] = x
        self.description['y'] = y
        self.description['text'] = text

    def observed(self, inputs, outputs):
        self.inputs = th.shared(inputs.astype(th.config.floatX))
        self.outputs = th.shared(outputs.astype(th.config.floatX))
        self.mean_inputs = self.mean(self.inputs)
        self.cov_inputs = self.kernel.cov(self.inputs)
        self.inv_outputs = self.mapping.inv(self.outputs)
        self.distribution = TGPDist('TGP', mu=self.mean_inputs, cov=self.cov_inputs, mapping=self.mapping,
                                    observed=self.outputs, testval=self.outputs, dtype=th.config.floatX)

    def testing(self, inputs, outputs):
        self.inputs_test = inputs.astype(th.config.floatX)
        self.outputs_test = outputs.astype(th.config.floatX)

    def swap_test_obs(self):
        swap = self.inputs.get_value()
        self.inputs.set_value(self.inputs_test)
        self.inputs_test = swap
        swap = self.outputs.get_value()
        self.outputs.set_value(self.outputs_test)
        self.outputs_test = swap

    def marginal_gp(self):
        value = tt.vector('marginal_gp')
        value.tag.test_value = zeros(1)
        delta = value - self.mean(self.space)
        cov = self.kernel.cov(self.space)
        cho = cholesky_robust(cov)
        L = sT.solve(cho, delta)
        return value, tt.exp(-np.float32(0.5) * (cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                                 + L.T.dot(L)) - tt.sum(tt.log(sT.diag(cho))))

    def marginal_tgp(self):
        value = tt.vector('marginal_tgp')
        value.tag.test_value = zeros(1)
        delta = self.mapping.inv(value) - self.mean(self.space)
        cov = self.kernel.cov(self.space)
        cho = cholesky_robust(cov)
        L = sT.solve(cho, delta)
        return value, tt.exp(-np.float32(0.5) * (cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                                 + L.T.dot(L)) - tt.sum(tt.log(sT.diag(cho))) + self.mapping.logdet_dinv(value))

    def prior_gp(self, cov=False):
        mu = self.mean(self.space)
        k_cov = self.kernel_f.cov(self.space)
        var = sT.extract_diag(k_cov)
        if cov:
            return mu, var, k_cov
        else:
            return mu, var

    def prior_tgp(self, sigma=2.0):
        mu, var = self.prior_gp(self.space)
        std = tt.sqrt(var)
        return self.mapping(mu), self.mapping(mu - sigma * std), self.mapping(mu + sigma * std)

    def prior_quantiles_tgp(self, sigma=1.96):
        mu, var = self.prior_gp()
        std = tt.sqrt(var)
        return self.mapping(mu), self.mapping(mu - sigma * std), self.mapping(mu + sigma * std)

    def prior_moments_tgp(self, n=20):
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX))
        w = th.shared(_w.astype(th.config.floatX))
        mu, var = self.prior_gp()
        std = tt.sqrt(var)
        return self.gauss_hermite(self.mapping, mu, std, a, w), self.gauss_hermite(lambda v: self.mapping(v) ** 2, mu,
                                                                                   std, a, w)

    def posterior_gp(self, cov=False):
        k_ni = self.kernel_f.cov(self.space, self.inputs)
        mu = self.mean(self.space) + k_ni.dot(sT.solve(self.cov_inputs, self.inv_outputs - self.mean_inputs))
        k_cov = self.kernel_f.cov(self.space) - k_ni.dot(sT.solve(self.cov_inputs, k_ni.T))
        var = sT.extract_diag(k_cov)
        if cov:
            return mu, var, k_cov
        else:
            return mu, var
        return mu, var

    def posterior_quantiles_tgp(self, sigma=1.96):
        mu, var = self.posterior_gp()
        std = tt.sqrt(var)
        return self.mapping(mu), self.mapping(mu - sigma * std), self.mapping(mu + sigma * std)

    def posterior_moments_tgp(self, n=20):
        _a, _w = np.polynomial.hermite.hermgauss(n)
        a = th.shared(_a.astype(th.config.floatX))
        w = th.shared(_w.astype(th.config.floatX))
        mu, var = self.posterior_gp()
        std = tt.sqrt(var)
        return self.gauss_hermite(self.mapping, mu, std, a, w), self.gauss_hermite(lambda v: self.mapping(v) ** 2, mu,
                                                                                   std, a, w)

    def sampler_gp(self):
        random = tt.vector('random')
        random.tag.test_value = zeros(self.space.shape.eval())
        if self.outputs is None:
            mu, var, cov = self.prior_gp(cov=True)
        else:
            mu, var, cov = self.posterior_gp(cov=True)
        cho = cholesky_robust(cov)
        return random, mu + random.dot(cho)

    def sampler_tgp(self):
        random, samples = self.sampler_gp()
        return random, self.mapping(samples)

    @staticmethod
    def gauss_hermite(f, mu, sigma, a, w):
        return tt.dot(w, f(mu + sigma.dimshuffle(['x', 0]) * np.sqrt(2) * a.dimshuffle([0, 'x']))) / np.sqrt(np.pi)

    def compile_functions(self):
        params = self.model.vars
        if self.outputs is None:
            mu, var = self.prior_gp()
            median, I_up, I_down = self.prior_quantiles_tgp()
            m1, m2 = self.prior_moments_tgp()
        else:
            mu, var = self.posterior_gp()
            median, I_up, I_down = self.posterior_quantiles_tgp()
            m1, m2 = self.posterior_moments_tgp()

        mu_compilated = makefn(params, mu)
        var_compilated = makefn(params, var)

        median_compilated = makefn(params, median)
        I_up_compilated = makefn(params, I_up)
        I_down_compilated = makefn(params, I_down)

        moment1 = makefn(params, m1)
        moment2 = makefn(params, m2)

        self.compiles['mean'] = mu_compilated
        self.compiles['variance'] = var_compilated
        self.compiles['median'] = median_compilated
        self.compiles['I_up'] = I_up_compilated
        self.compiles['I_down'] = I_down_compilated
        self.compiles['m1'] = moment1
        self.compiles['m2'] = moment2
        return mu_compilated, var_compilated, median_compilated, I_up_compilated, I_down_compilated, moment1, moment2

    def compile_samplers(self):
        params = self.model.vars
        random, sampler = self.sampler_gp()
        sampler_gp = makefn([random] + params, sampler)
        random, sampler = self.sampler_tgp()
        sampler_tgp = makefn([random] + params, sampler)
        self.compiles['sampler_gp'] = sampler_gp
        self.compiles['sampler_tgp'] = sampler_tgp
        return sampler_gp, sampler_tgp

    def compile_distributions(self, index=0):
        params = self.model.vars
        value, dist = self.marginal_gp()

        dist_gp = makefn([value] + params, dist[index])

        trans = makefn([value] + params, self.mapping(value))
        trans_inv = makefn([value] + params, self.mapping.inv(value))
        det_jac_trans_inv = makefn([value] + params, tt.exp(self.mapping.logdet_dinv(value)))

        value, dist = self.marginal_tgp()
        dist_tgp = makefn([value] + params, dist[index])

        self.compiles['trans'] = trans
        self.compiles['trans_inv'] = trans_inv
        self.compiles['det_jac_trans_inv'] = det_jac_trans_inv
        self.compiles['dist_gp'] = dist_gp
        self.compiles['dist_tgp'] = dist_tgp
        return trans, trans_inv, det_jac_trans_inv, dist_gp, dist_tgp

    def compile(self, index=0):
        self.compile_functions()
        self.compile_samplers()
        #self.compile_distributions(index)

    def plot_data(self, big=False):
        if big:
            self.plot_data_big()
        else:
            self.plot_data_normal()

    def plot_data_normal(self):
        try:
            if self.hidden is not None:
                plt.plot(self.space_x, self.hidden, label='Hidden Processes')
        except:
            pass
        plt.plot(self.inputs.get_value(), self.outputs.get_value(), '.k', label='Observations')

    def plot_data_big(self):
        try:
            if self.hidden is not None:
                plt.plot(self.space_x, self.hidden, linewidth=4, label='Hidden Processes')
        except:
            pass
        plt.plot(self.inputs.get_value(), self.outputs.get_value(), '.k', ms=20, label='Observations')

    def plot_gp_moments(self, params):
        mean = self.compiles['mean'](**params)
        variance = self.compiles['variance'](**params)
        std = np.sqrt(variance)
        plt.plot(self.space_x, mean, label='Mean')
        plt.fill_between(self.space_x, mean + 1.96 * std, mean - 1.96 * std, alpha=0.2, label='95%')
        return mean, variance

    def plot_gp_samples(self, params, n=1):
        for i in range(n):
            plt.plot(self.space_x, self.compiles['sampler_gp'](np.random.randn(len(self.space_x)), **params))

    def plot_tgp_quantiles(self, params):
        med = self.compiles['median'](**params)
        Iup = self.compiles['I_up'](**params)
        Idown = self.compiles['I_down'](**params)
        plt.plot(self.space_x, med, label='Median')
        plt.fill_between(self.space_x, Iup, Idown, alpha=0.2, label='95%')
        return med, Iup, Idown

    def plot_tgp_moments(self, params):
        moment1 = self.compiles['m1'](**params)
        moment2 = self.compiles['m2'](**params)
        std = np.sqrt(moment2-moment1**2)
        plt.plot(self.space_x, moment1, label='Mean')
        #plt.plot(self.space_x, moment1 + 1.96 * std, '--k', alpha=0.2, label='3.92 std')
        #plt.plot(self.space_x, moment1 - 1.96 * std, '--k', alpha=0.2)
        return moment1, moment2

    def plot_tgp_samples(self, params, n=1):
        for i in range(n):
            plt.plot(self.space_x, self.compiles['sampler_tgp'](np.random.randn(len(self.space_x)), **params))

    def plot_gp(self, params, title=None, samples=0, big=True, loc=1):
        self.plot_data(big)
        self.plot_gp_moments(params)
        self.plot_gp_samples(params, samples)
        self.description['scores'] = self.scores(params)
        score_title = 'll_train={:.2f} | ll_test={:.2f}'.format(self.description['scores']['logp_train'],
                                                                self.description['scores']['logp_test'])
        if title is None:
            title = self.description['title']
        text_plot(title + ': ' + score_title,  self.description['x'],  self.description['y'], loc=loc)

    def plot_tgp(self, params, title=None, samples=0, big=True, scores=True, loc=1):
        self.plot_data(big)
        self.plot_tgp_moments(params)
        self.plot_tgp_quantiles(params)
        self.plot_tgp_samples(params, samples)
        self.description['scores'] = self.scores(params)
        if scores:
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

    def scores(self, params):
        try:
            logp_train = np.nanmean(self.model.logp(params))
        except:
            logp_train = np.float32(0)

        mean = self.compiles['m1'](**params)
        var = mean**2 - self.compiles['m2'](**params)
        #print(mean.shape)
        #print(self.outputs.get_value().shape)

        mse_train = 0#np.nanmean((mean - self.outputs.get_value()) ** 2 + var)
        bias_train = 0#np.nanmean(np.abs(mean - self.outputs.get_value()))

        self.swap_test_obs()
        try:
            logp_test = np.nanmean(self.model.logp(params))
        except:
            logp_test = np.float32(0)
        mean = self.compiles['m1'](**params)
        var = mean**2 - self.compiles['m2'](**params)
        mse_test = 0#np.nanmean((mean - self.outputs.get_value()) ** 2 + var)
        bias_test = 0#np.nanmean(np.abs(mean - self.outputs.get_value()))
        self.swap_test_obs()

        d = {'logp_train': logp_train,
             'mse_train': mse_train,
             'mab_train': bias_train,
             'logp_test': logp_test,
             'mse_test': mse_test,
             'mab_test': bias_test}
        return d

    def find_MAP(self, start, num=1, plot=False):
        maps = list()
        maps.append(('start', self.model.logp(start), start))
        with self.model:
            for i in range(num):
                try:
                    name, logp, start = maps[i // 2]
                    if i % 2 == 0:
                        name += '_bfgs'
                        new = pm.find_MAP(fmin=sp.optimize.fmin_bfgs, start=start)
                    else:
                        name += '_powell'
                        new = pm.find_MAP(fmin=sp.optimize.fmin_powell, start=start)
                    maps.append((name, self.model.logp(new), new))
                except:
                    pass
        if plot:
            n = 0
            for name, logp, params in maps:
                plt.figure(n)
                self.plot_tgp(params, name)
                n += 1
        return maps

    def default_hypers(self):
        x = self.inputs.get_value()
        y = self.outputs.get_value()
        return {**self.kernel.default_hypers(x, y), **self.mapping.default_hypers(x, y), **self.mean.default_hypers(x, y)}

    def find_default(self):
        default = {}
        for k, v in trans_hypers(self.default_hypers()).items():
            default[k.name] = v
        return default

    def plot_tgp_widget(self, **params):
        self.plot_data(big=True)
        self.plot_tgp_quantiles(params)
       # self.plot_tgp_moments(params)
        self.description['scores'] = self.scores(params)
        text_plot(self.description['title'],  self.description['x'],  self.description['y'], loc=1)
        self._widget_params = params

    def show_widget(self, params=None):
        if params is None:
            params = self.widget_params()
        intervals = dict()
        for k, v in params.items():
            if v > 0:
                intervals[k] = [0, 2*v]
            elif v < 0:
                intervals[k] = [2*v, 0]
            else:
                intervals[k] = [-1.99, 1.99]
        widgets.interact(self.plot_tgp_widget, __manual=True, **intervals)

    def widget_params(self):
        if self._widget_params is None:
            return self.find_default()
        return self._widget_params

    def sample_hypers(self, start, samples=100, chains=1):
        pm.Slice
        with self.model:
            advi_mu, advi_sm, advi_elbo = pm.advi(start=start, n=samples)
            step = pm.HamiltonianMC(scaling=advi_sm)
            trace = pm.sample(samples, step=step, start=advi_mu, njobs=chains)
        return trace

    def get_point(self, trace, i):
        return {v.name: trace[i][v.name] for v in self.model.vars}

    def plot_cov(self):
        return plt.matshow(self.cov_inputs)


class TGPDist(pm.Continuous):
    def __init__(self, mu, cov, mapping, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov
        self.mapping = mapping

    def logp_cov(self, value):  # es más rápido pero se cae
        delta = self.mapping.inv(value) - self.mu
        return -np.float32(0.5) * (tt.log(sT.det(self.cov)) + delta.T.dot(sT.solve(self.cov, delta))
                                   + self.cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))) \
               + self.mapping.logdet_dinv(value)

    def logp_cho(self, value):
        delta = self.mapping.inv(value) - self.mu
        L = sT.solve(self.cho, delta)
        return -np.float32(0.5) * (self.cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                   + L.T.dot(L)) - tt.sum(tt.log(sT.diag(self.cho))) + self.mapping.logdet_dinv(value)

    def logp(self, value):
        return self.logp_cho(value)

    @property
    def cho(self):
        return cholesky_robust(self.cov)


class GP(TGP):
    def __init__(self, space, mean, kernel, noise=True, name=None, hidden=None):
        super().__init__(space, mean, kernel, Identity(), noise=noise, name=name, hidden=hidden)
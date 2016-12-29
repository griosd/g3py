import pickle
import numpy as np
import pymc3 as pm
import scipy as sp
import theano as th
from g3py import Mean, Kernel, Mapping, Model, KernelSum, WN, tt_to_cov, tt_to_num, TGPDist, zeros, cholesky_robust, \
    def_space, debug, makefn, text_plot, trans_hypers, ConstantStep, RobustSlice, Identity
from ipywidgets import interact
from matplotlib import pyplot as plt
from theano import tensor as tt
from theano.tensor import slinalg as sL
from theano.tensor import nlinalg as nL


class TGP:
    def __init__(self, space, mean: Mean, kernel: Kernel, mapping: Mapping, noise=True, name=None, hidden=None, description=None):
        self.model = Model.get_context()
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.space, self.space_x, self.space_t = def_space(space, self.name + '_Space')

        self.mean = mean
        if noise:
            self.kernel_f = kernel
            self.kernel = KernelSum(self.kernel_f, WN(self.space_x, 'Noise'))
        else:
            self.kernel_f = kernel
            self.kernel = self.kernel_f
        self.mapping = mapping

        self.inputs = None
        self.inputs_x = None
        self.inputs_t = None
        self.outputs = None
        self.outputs_y = None
        self.mean_inputs = None
        self.cov_inputs = None
        self.inv_outputs = None
        self.distribution = None
        self.inputs_test = None
        self.outputs_test = None
        self.fixed_params = {}
        self._widget_params = {}
        self._widget_trace = None
        self.compiles = {}
        if description is None:
            self.description = {'title': 'title',
                                'x': 'x',
                                'y': 'y',
                                'text': 'text'}
        else:
            self.description = description
        if hidden is None:
            self.hidden = None
        else:
            self.hidden = np.squeeze(hidden)
        with self.model:
            self.check_hypers()

    def check_hypers(self):
        self.mean.check_hypers(self.name+'_')
        self.kernel.check_hypers(self.name+'_')
        self.mapping.check_hypers(self.name+'_')

    def set_space(self, space):
        new_space, self.space_x, self.space_t = def_space(space, self.name + '_Space')
        self.space.set_value(new_space.get_value())

    def describe(self, title, x, y, text=''):
        self.description['title'] = title
        self.description['x'] = x
        self.description['y'] = y
        self.description['text'] = text

    def observed(self, inputs, outputs):
        self.inputs, self.inputs_x, self.inputs_t = def_space(inputs, self.name + '_inputs')
        self.outputs, self.outputs_y, __ = def_space(outputs, self.name + '_outputs', squeeze=True)
        with Model.get_context():
            self.mean_inputs = self.mean(self.inputs)
            self.cov_inputs = tt_to_cov(self.kernel.cov(self.inputs))
            self.inv_outputs = tt_to_num(self.mapping.inv(self.outputs))
            self.distribution = TGPDist(self.name, mu=self.mean_inputs, cov=self.cov_inputs, mapping=self.mapping,
                                        tgp=self, observed=self.outputs, testval=self.outputs, dtype=th.config.floatX)

    # TODO
    def testing(self, inputs, outputs):
        __, self.inputs_test, __ = def_space(inputs)
        __, self.outputs_test, __ = def_space(outputs, squeeze=True)

    def point_to_eval(self, point):
        r = dict()
        for k, v in point.items():
            r[self.model[k]] = v
        return r

    def point_to_In(self, point):
        r = list()
        for k, v in point.items():
            r.append(th.In(self.model[k], value=v))
        return r

    # TODO
    def swap_test_obs(self):
        if self.inputs is None:
            return
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
        L = sL.solve_lower_triangular(cho, delta)
        return value, tt.exp(-np.float32(0.5) * (cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                                 + L.T.dot(L)) - tt.sum(tt.log(nL.extract_diag(cho))))

    def marginal_tgp(self):
        value = tt.vector('marginal_tgp')
        value.tag.test_value = zeros(1)
        delta = self.mapping.inv(value) - self.mean(self.space)
        cov = self.kernel.cov(self.space)
        cho = cholesky_robust(cov)
        L = sL.solve_lower_triangular(cho, delta)
        return value, tt.exp(-np.float32(0.5) * (cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                                 + L.T.dot(L)) - tt.sum(tt.log(nL.extract_diag(cho))) + self.mapping.logdet_dinv(value))

    def prior_gp(self, cov=False, noise=False):
        mu = self.mean(self.space)
        if noise:
            k_cov = self.kernel.cov(self.space)
        else:
            k_cov = self.kernel_f.cov(self.space)
        var = nL.extract_diag(k_cov)
        if cov:
            return mu, var, k_cov
        else:
            return mu, var

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

    def posterior_gp(self, cov=False, noise=False):
        k_ni = self.kernel_f.cov(self.space, self.inputs)
        mu = self.mean(self.space) + k_ni.dot(sL.solve(self.cov_inputs, self.inv_outputs - self.mean_inputs))
        if noise:
            k_cov = self.kernel.cov(self.space) - k_ni.dot(sL.solve(self.cov_inputs, k_ni.T))
        else:
            k_cov = self.kernel_f.cov(self.space) - k_ni.dot(sL.solve(self.cov_inputs, k_ni.T))
        var = nL.extract_diag(debug(k_cov, 'k_cov'))
        if cov:
            return mu, var, k_cov
        else:
            return mu, var

    def subprocess_gp(self, subkernel, cov=False, noise=False):
        k_ni = subkernel.cov(self.space, self.inputs)
        mu = self.mean(self.space) + k_ni.dot(sL.solve(self.cov_inputs, self.inv_outputs - self.mean_inputs))
        if noise:
            k_cov = self.kernel.cov(self.space) - k_ni.dot(sL.solve(self.cov_inputs, k_ni.T))
        else:
            k_cov = self.kernel_f.cov(self.space) - k_ni.dot(sL.solve(self.cov_inputs, k_ni.T))
        var = nL.extract_diag(debug(k_cov, 'k_cov'))
        if cov:
            return mu, var, k_cov
        else:
            return mu, var

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

    def sampler_gp(self):
        random = tt.vector('random')
        random.tag.test_value = zeros(self.space_x.shape[0])
        if self.outputs is None:
            mu, var, cov = self.prior_gp(cov=True)
        else:
            mu, var, cov = self.posterior_gp(cov=True)
        cho = cholesky_robust(cov)
        return random, mu + cho.dot(random)

    def sampler_tgp(self):
        random, samples = self.sampler_gp()
        return random, self.mapping(samples)

    @staticmethod
    def gauss_hermite(f, mu, sigma, a, w):
        return tt.dot(w, f(mu + sigma * np.sqrt(2) * a)) / np.sqrt(np.pi)

    def compile_functions(self):
        params = self.model.vars
        if self.outputs is None:
            mu, var, cov = self.prior_gp(cov=True)
            _, noise = self.prior_gp(cov=False, noise=True)
            median, I_up, I_down = self.prior_quantiles_tgp()
            _, noise_up, noise_down = self.prior_quantiles_tgp(noise=True)
            m1, m2 = self.prior_moments_tgp()
        else:
            mu, var, cov = self.posterior_gp(cov=True)
            _, noise = self.posterior_gp(cov=False, noise=True)
            median, I_up, I_down = self.posterior_quantiles_tgp()
            _, noise_up, noise_down = self.posterior_quantiles_tgp(noise=True)
            m1, m2 = self.posterior_moments_tgp()

        mu_compilated = makefn(params, mu)
        var_compilated = makefn(params, var)
        cov_compilated = makefn(params, cov)
        noise_compilated = makefn(params, noise)

        median_compilated = makefn(params, median)
        I_up_compilated = makefn(params, I_up)
        I_down_compilated = makefn(params, I_down)

        noise_up_compilated = makefn(params, noise_up)
        noise_down_compilated = makefn(params, noise_down)

        moment1 = makefn(params, m1)
        moment2 = makefn(params, m2)

        self.compiles['mean'] = mu_compilated
        self.compiles['variance'] = var_compilated
        self.compiles['covariance'] = cov_compilated
        self.compiles['noise'] = noise_compilated
        self.compiles['median'] = median_compilated
        self.compiles['I_up'] = I_up_compilated
        self.compiles['I_down'] = I_down_compilated
        self.compiles['noise_up'] = noise_up_compilated
        self.compiles['noise_down'] = noise_down_compilated
        self.compiles['m1'] = moment1
        self.compiles['m2'] = moment2
        return mu_compilated, var_compilated, noise_compilated, median_compilated, I_up_compilated, I_down_compilated, \
               noise_up_compilated, noise_down_compilated, moment1, moment2

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

    def plot_space(self):
        plt.plot(self.space_t, self.space_x)
        if self.inputs_t is not None:
            plt.plot(self.inputs_t, self.inputs_x)

    def plot_data(self, big=False):
        if big:
            self.plot_data_big()
        else:
            self.plot_data_normal()

    def plot_data_normal(self):
        if self.hidden is not None:
            plt.plot(self.space_t, self.hidden, label='Hidden Processes')
        if self.outputs_y is not None:
            plt.plot(self.inputs_t, self.outputs_y, '.k', label='Observations')

    def plot_data_big(self):
        if self.hidden is not None:
            plt.plot(self.space_t, self.hidden, linewidth=4, label='Hidden Processes')
        if self.outputs_y is not None:
            plt.plot(self.inputs_t, self.outputs_y, '.k', ms=20, label='Observations')

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

    def predict_gp(self, params, mean=True, var=True, cov=False, noise=False):
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

    def sample_gp(self, params, samples=1):
        S = np.empty((len(self.space_x), samples))
        for i in range(samples):
            S[:, i] = self.compiles['sampler_gp'](np.random.randn(len(self.space_x)), **params)
        return S

    def plot_gp_samples(self, params, samples=1):
        for i in range(samples):
            plt.plot(self.space_t, self.compiles['sampler_gp'](np.random.randn(len(self.space_x)), **params), alpha=0.6)

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

    def find_MAP(self, start=None, points=1, plot=False, return_points=False, display=True):
        points_list = list()
        if start is None:
            start = self.find_default()
        points_list.append(('start', self.model.logp(start), start))
        plt.figure(0)
        print('Starting function value (-logp): '+str(-self.model.logp(start)))
        self.plot_tgp(start, 'start')
        plt.show()
        with self.model:
            for i in range(points):
                try:
                    name, logp, start = points_list[i // 2]
                    if i % 2 == 0:
                        name += '_bfgs'

                        new = pm.find_MAP(fmin=sp.optimize.fmin_bfgs, vars=self.sampling_vars, start=start)
                    else:
                        name += '_powell'
                        print(name)
                        new = pm.find_MAP(fmin=sp.optimize.fmin_powell, vars=self.sampling_vars, start=start)
                    points_list.append((name, self.model.logp(new), new))
                    if plot:
                        plt.figure(i+1)
                        self.plot_tgp(new, name)
                        plt.show()
                except:
                    pass
        optimal = points_list[0]
        for test in points_list:
            if test[1] > optimal[1]:
                optimal = test
        name, logp, params = optimal
        if display:
            #print(params)
            pass
        if return_points is False:
            return params
        else:
            return params, points_list

    def default_hypers(self):
        x = self.inputs.get_value()
        y = self.outputs.get_value()
        return {**self.kernel.default_hypers_dims(x, y), **self.mapping.default_hypers_dims(x, y),
                **self.mean.default_hypers_dims(x, y)}

    def find_default(self):
        if self.inputs is None:
            return self.model.test_point
        default = {}
        for k, v in trans_hypers(self.default_hypers()).items():
            default[k.name] = v
        default.update(self.fixed_params)
        return default

    def eval_default(self):
        return self.point_to_eval(self.find_default())

    def plot_model(self, **params):
        plt.figure(2)
        _ = plt.plot(self.compiles['covariance'](**params)[len(self.space_x) // 2, :])
        text_plot('kernel', 'x1', 'x2')
        return

    def check_params_dims(self, **params):
        r = dict()
        for k, v in params.items():
            r[k] = np.array(v, dtype=th.config.floatX).reshape(self.model[k].tag.test_value.shape)
        return r

    def plot_tgp_widget(self, **params):
        params = self.check_params_dims(**params)
        self._widget_params = params
        self.plot_data(big=True)
        self.plot_tgp_quantiles(params)
        self.plot_tgp_moments(params)
        #self.description['scores'] = self.scores(params)
        text_plot(self.description['title'],  self.description['x'],  self.description['y'], loc=1)
        #self.plot_model(**params)

    def widget_params(self, params=None):
        if params is None:
            params = self.get_params()
        intervals = dict()
        for k, v in params.items():
            v = np.squeeze(v)
            if v > 0.1:
                intervals[k] = [0, 2*v]
            elif v < -0.1:
                intervals[k] = [2*v, 0]
            else:
                intervals[k] = [-5.00, 5.00]
        interact(self.plot_tgp_widget, __manual=True, **intervals)

    def get_params(self):
        if len(self._widget_params) == 0:
            return self.find_default()
        self._widget_params.update(self.fixed_params)
        return self._widget_params

    def get_point(self, trace, i):
        return {v.name: trace[i][v.name] for v in self.model.vars}

    def point(self, point):
        return {v.name: point[v.name] for v in self.model.vars}

    def plot_trace(self, id_trace=0):
        self._widget_params = self.get_point(self._widget_trace, id_trace)
        self.plot_tgp_widget(**self._widget_params)

    def widget_trace(self, trace):
        self._widget_trace = trace
        interact(self.plot_trace, id_trace=[0, len(self._widget_trace) - 1])

    def plot_tgp_trace(self, trace):
        self.plot_data(big=True)
        self.plot_tgp_quantiles(params)
       # self.plot_tgp_moments(params)
        self.description['scores'] = self.scores(params)
        text_plot(self.description['title'],  self.description['x'],  self.description['y'], loc=1)
        self._widget_params = params

    def plot_cov(self):
        return plt.matshow(self.cov_inputs)

    def fix_params(self, fixed_params):
        self.fixed_params = fixed_params

    @property
    def fixed_vars(self):
        return [t for t in self.model.vars if t.name in self.fixed_params.keys()]

    @property
    def sampling_vars(self):
        return [t for t in self.model.vars if t not in self.fixed_vars]

    def sample_hypers(self, start=None, samples=1000, chains=1, advi=True):
        if start is None:
            start = self.find_default()
        with self.model:
            if advi:
                advi_mu, advi_sm, advi_elbo = pm.advi(vars=self.sampling_vars, start=start, n=20000) #OK
                for k, v in advi_sm.items():
                    advi_sm[k] = v**2
            else:
                advi_mu = start
                advi_sm = None
            if len(self.fixed_params) > 0:
                step = [ConstantStep(vars=self.fixed_vars)]  # Fue eliminado por error en pymc3
                step += [RobustSlice(vars=self.sampling_vars)]  # Slice original se cuelga si parte del óptimo
                #step += [pm.Metropolis(vars=self.sampling_vars, tune=False)] # OK
                #step += [pm.HamiltonianMC(vars=self.sampling_vars, scaling=advi_sm, is_cov=True)] #Scaling is not positive definite. Simple check failed. Diagonal contains negatives. Check indexes
                #step += [pm.NUTS(vars=self.sampling_vars, scaling=advi_sm, is_cov=True)] #BUG float32
            else:
                step = RobustSlice()
            trace = pm.sample(samples, step=step, start=advi_mu, njobs=chains)
        return trace

    def save_model(self, path, params=None):
        try:
            with self.model:
                with open(path, 'wb') as f:
                    pickle.dump((params), f, protocol=-1)
            print('Saved model '+path)
        except:
            print('Error saving model '+path)

class GP(TGP):
    def __init__(self, space, mean, kernel, noise=True, name=None, hidden=None):
        super().__init__(space, mean, kernel, Identity(), noise=noise, name=name, hidden=hidden)


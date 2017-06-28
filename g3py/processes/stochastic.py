import numpy as np
import theano as th
import theano.tensor as tt
from ..libs.tensors import tt_to_num, tt_to_cov, cholesky_robust
#from ..bayesian.models import TheanoBlackBox
from .hypers import Freedom
from .hypers.means import Mean
from .hypers.kernels import Kernel, KernelSum, WN
from .hypers.mappings import Mapping, Identity
from ..bayesian.models import GraphicalModel
import theano.tensor.slinalg as tsl
from ..libs.plots import figure, plot, show, plot_text, grid2d, plot_2d
#import theano.tensor.nlinalg as tnl
from .. import config
import matplotlib.pyplot as plt
from ipywidgets import interact
from matplotlib import cm


class StochasticProcess:#TheanoBlackBox

    def __init__(self, name='SP', space=None, index=None, inputs=None, outputs=None, hidden=None, distribution=None, description=None):
        self.name = name
        self.th_space = space
        self.th_index = index
        self.th_inputs = inputs
        self.th_outputs = outputs
        self.th_hidden = hidden

        if self.th_order is None:
            self.th_order = tt.vector(self.name + '_order', dtype=th.config.floatX)
            self.th_order.tag.test_value = np.array([0.0, 1.0], dtype=th.config.floatX)

        if self.th_space is None:
            self.th_space = tt.matrix(self.name + '_space', dtype=th.config.floatX)
            self.th_space.tag.test_value = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=th.config.floatX)

        if self.th_hidden is None:
            self.th_hidden = tt.vector(self.name + '_hidden', dtype=th.config.floatX)
            self.th_hidden.tag.test_value = np.array([0.0, 1.0], dtype=th.config.floatX)

        if self.th_index is None:
            self.th_index = tt.vector(self.name + '_index', dtype=th.config.floatX)
            self.th_index.tag.test_value = np.array([0.0, 1.0], dtype=th.config.floatX)

        if self.th_inputs is None:
            self.th_inputs = tt.matrix(self.name + '_inputs', dtype=th.config.floatX)
            self.th_inputs.tag.test_value = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=th.config.floatX)

        if self.th_outputs is None:
            self.th_outputs = tt.vector(self.name + '_outputs', dtype=th.config.floatX)
            self.th_space.tag.test_value = np.array([0.0, 1.0], dtype=th.config.floatX)

        self.distribution = distribution
        self.description = description
        if GraphicalModel.active is None:
            GraphicalModel.active = GraphicalModel('GM_'+self.name, description=description)
        self.model = GraphicalModel.active

    def _define_process(self, distribution = None):
        self.distribution = distribution

    def default_hypers(self):
        pass

    def _check_process(self):
        pass

    def quantiler(self, q=0.975, prior=False, noise=False):
        pass

    def sampler(self, nsamples=1, prior=False):
        pass

    def mean(self):
        pass

    def variance(self):
        pass

    def covariance(self):
        pass

    def predictive(self):
        pass

    def median(self):
        pass

    def std(self, *args, **kwargs):
        return tt.sqrt(self.variance(*args, **kwargs))

    def describe(self, title=None, x=None, y=None, text=None):
        if title is not None:
            self.description['title'] = title
        if title is not None:
            self.description['x'] = x
        if title is not None:
            self.description['y'] = y
        if title is not None:
            self.description['text'] = text

    def plot_space(self, space=None, independ=False ,observed=False):
        if space is not None:
            self.set_space(space)
        if independ:
            for i in range(self.space_values.shape[1]):
                figure(i)
                plot(self.th_order, self.space_values[:, i])
        else:
            plot(self.th_order, self.space_values)
        if self.th_index is not None and observed:
            if independ:
                for i in range(self.space_values.shape[1]):
                    figure(i)
                    plot(self.th_index, self.inputs_values[:, i], '.k')
            else:
                plot(self.th_index, self.inputs_values, '.k')

    def plot_hidden(self, big=None):
        if big is None:
            big = config.plot_big
        if big and self.hidden is not None:
            plot(self.th_order, self.hidden[0:len(self.th_order)], linewidth=4, label='Hidden Processes')
        elif self.hidden is not None:
            plot(self.th_order, self.hidden[0:len(self.th_order)],  label='Hidden Processes')

    def plot_observations(self, big=None):
        if big is None:
            big = config.plot_big
        if big and self.th_outputs is not None:
            plot(self.th_index, self.th_outputs, '.k', ms=20)
            plot(self.th_index, self.th_outputs, '.r', ms=15, label='Observations')
        elif self.th_outputs is not None:
            plot(self.th_index, self.th_outputs, '.k', ms=10)
            plot(self.th_index, self.th_outputs, '.r', ms=6, label='Observations')

    def plot(self, params=None, space=None, inputs=None, outputs=None, mean=True, var=False, cov=False, median=False, quantiles=True, noise=True, samples=0, prior=False,
             data=True, big=None, plot_space=False, title=None, loc=1):
        values = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, samples=samples, prior=prior)
        if space is not None:
            self.set_space(space)
        if data:
            self.plot_hidden(big)
        if mean:
            plot(self.th_order, values['mean'], label='Mean')
        if var:
            plot(self.th_order, values['mean'] + 2.0 * values['std'], '--k', alpha=0.2, label='4.0 std')
            plot(self.th_order, values['mean'] - 2.0 * values['std'], '--k', alpha=0.2)
        if cov:
            pass
        if median:
            plot(self.th_order, values['median'], label='Median')
        if quantiles:
            plt.fill_between(self.th_order, values['quantile_up'], values['quantile_down'], alpha=0.1, label='95%')
        if noise:
            plt.fill_between(self.th_order, values['noise_up'], values['noise_down'], alpha=0.1, label='noise')
        if samples > 0:
            plot(self.th_order, values['samples'], alpha=0.4)
        if title is None:
            title = self.description['title']
        if data:
            self.plot_observations(big)
        if loc is not None:
            plot_text(title, self.description['x'], self.description['y'], loc=loc)
        if plot_space:
            show()
            self.plot_space()
            plot_text('Space X', 'Index', 'Value', legend=False)

    def plot_distribution(self, index=0, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, prior=False, sigma=4, neval=100, title=None, swap=False, label=None):
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, distribution=True, prior=prior)
        domain = np.linspace(pred.mean - sigma * pred.std, pred.mean + sigma * pred.std, neval)
        dist_plot = np.zeros(len(domain))
        for i in range(len(domain)):
            dist_plot[i] = pred.distribution(domain[i:i + 1])
        if label is None:
            if prior:
                label='prior'
            else:
                label='posterior'
        if label is False:
            label = None
        if title is None:
            title = 'Marginal Distribution y_'+str(self.th_order[index])
        if swap:
            plot(dist_plot, domain, label=label)
            plot_text(title, 'Density', 'Domain y')
        else:
            plot(domain, dist_plot,label=label)
            plot_text(title, 'Domain y', 'Density')

    def plot_distribution2D(self, indexs=[0, 1], params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, prior=False, sigma_1=2, sigma_2=2, neval=33, title=None):
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, distribution=True, prior=prior)
        dist1 = np.linspace(pred.mean[0] - sigma_1 * pred.std[0], pred.mean[0] + sigma_1 * pred.std[0], neval)
        dist2 = np.linspace(pred.mean[1] - sigma_2 * pred.std[1], pred.mean[1] + sigma_2 * pred.std[1], neval)
        xy, x2d, y2d = grid2d(dist1, dist2)
        dist_plot = np.zeros(len(xy))
        for i in range(len(xy)):
            dist_plot[i] = pred.distribution(xy[i])
        plot_2d(dist_plot, x2d, y2d)
        if title is None:
            title = 'Distribution2D'
        plot_text(title, 'Domain y_'+str(self.th_order[indexs[0]]), 'Domain y_'+str(self.th_order[indexs[1]]), legend=False)

    def plot_model(self, params=None, indexs=None, kernel=True, mapping=True, marginals=True, bivariate=True):
        if indexs is None:
            indexs = [self.th_index[len(self.th_index)//2], self.th_index[len(self.th_index)//2]+1]

        if kernel:
            plt.subplot(121)
            self.plot_kernel(params=params)
        if mapping:
            plt.subplot(122)
            self.plot_mapping(params=params)
        show()

        if marginals:
            plt.subplot(121)
            self.plot_distribution(index=indexs[0], params=params, space=self.space_values[indexs[0]:indexs[0]+1, :], prior=True)
            self.plot_distribution(index=indexs[0], params=params, space=self.space_values[indexs[0]:indexs[0]+1, :])
            plt.subplot(122)
            self.plot_distribution(index=indexs[1], params=params, space=self.space_values[indexs[1]:indexs[1]+1, :], prior=True)
            self.plot_distribution(index=indexs[1], params=params, space=self.space_values[indexs[1]:indexs[1]+1, :])
            show()
        if bivariate:
            self.plot_distribution2D(indexs=indexs, params=params, space=self.space_values[indexs, :])
            show()

    def plot_kernel2D(self):
        pass

    def _check_params_dims(self, params):
        r = dict()
        for k, v in params.items():
            try:
                r[k] = np.array(v, dtype=th.config.floatX).reshape(self.model[k].tag.test_value.shape)
            except KeyError:
                pass
        return r

    def _widget_plot(self, params):
        self.params_widget = params
        self.plot(params=self.params_widget, samples=self._widget_samples)

    def _widget_plot_trace(self, id_trace):
        self._widget_plot(self._check_params_dims(self._widget_traces[id_trace]))

    def _widget_plot_params(self, **params):
        self._widget_plot(self._check_params_dims(params))

    def _widget_plot_model(self, **params):
        self.params_widget = self._check_params_dims(params)
        self.plot_model(params=self.params_widget, indexs=None, kernel=False, mapping=True, marginals=True,
                        bivariate=False)

    def widget_traces(self, traces, chain=0):
        self._widget_traces = traces._straces[chain]
        interact(self._widget_plot_trace, __manual=True, id_trace=[0, len(self._widget_traces) - 1])

    def widget_params(self, params=None, samples=0):
        if params is None:
            params = self.get_params_widget()
        intervals = dict()
        for k, v in params.items():
            v = np.squeeze(v)
            if v > 0.1:
                intervals[k] = [0, 2*v]
            elif v < -0.1:
                intervals[k] = [2*v, 0]
            else:
                intervals[k] = [-5.00, 5.00]
        self._widget_samples = samples
        interact(self._widget_plot_params, __manual=True, **intervals)

    def widget_model(self, params=None):
        if params is None:
            params = self.get_params_widget()
        intervals = dict()
        for k, v in params.items():
            v = np.squeeze(v)
            if v > 0.1:
                intervals[k] = [0, 2*v]
            elif v < -0.1:
                intervals[k] = [2*v, 0]
            else:
                intervals[k] = [-5.00, 5.00]
        interact(self._widget_plot_model, __manual=True, **intervals)


class EllipticalProcess(StochasticProcess):
    def __init__(self, location: Mean=None, kernel: Kernel=None, degree: Freedom=None, mapping: Mapping=Identity(),
                 noise=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.location = location
        self.degree = degree
        self.mapping = mapping

        if noise:
            self.kernel_f = kernel
            self.kernel = KernelSum(self.kernel_f, WN(self.space_values, 'Noise'))
        else:
            self.kernel_f = kernel
            self.kernel = self.kernel_f

    def default_hypers(self):
        x = np.array(self.inputs_values)
        y = np.squeeze(np.array(self.outputs_values))
        return {**self.location.default_hypers_dims(x, y), **self.kernel.default_hypers_dims(x, y),
                **self.mapping.default_hypers_dims(x, y)}

    def _define_process(self):
        # Basic Tensors
        self.prior_location_space = self.location(self.th_space)
        self.prior_location_inputs = self.location(self.th_inputs)

        self.prior_kernel_space = tt_to_cov(self.kernel.cov(self.th_space))
        self.prior_kernel_inputs = tt_to_cov(self.kernel.cov(self.th_inputs))
        self.prior_cholesky_space = cholesky_robust(self.prior_kernel_space)

        self.prior_kernel_f_space = self.kernel_f.cov(self.th_space)
        self.prior_kernel_f_inputs = self.kernel_f.cov(self.th_inputs)
        self.prior_cholesky_f_space = cholesky_robust(self.prior_kernel_f_space)

        self.cross_kernel_space_inputs = tt_to_num(self.kernel.cov(self.th_space, self.th_inputs))
        self.cross_kernel_f_space_inputs = tt_to_num(self.kernel_f.cov(self.th_space, self.th_inputs))

        self.posterior_location_space = self.prior_mean + self.cross_kernel_f_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.mapping_outputs - self.prior_location_inputs))

        self.posterior_kernel_space = self.prior_kernel_space - self.cross_kernel_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.cross_kernel_space_inputs.T))
        self.posterior_cholesky_space = cholesky_robust(self.posterior_kernel_space)

        self.posterior_kernel_f_space = self.prior_kernel_f_space - self.cross_kernel_f_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.cross_kernel_f_space_inputs.T))
        self.posterior_cholesky_f_space = cholesky_robust(self.posterior_kernel_f_space)

        self.mapping_outputs = tt_to_num(self.mapping.inv(self.th_outputs))
        #self.mapping_th = tt_to_num(self.mapping(self.random_th))
        #self.mapping_inv_th = tt_to_num(self.mapping.inv(self.random_th))

    def location(self, prior=False):
        if prior:
            return self.prior_location_space
        else:
            return self.posterior_location_space

    def kernel(self, prior=False, noise=False):
        if prior:
            if noise:
                return self.prior_kernel_space
            else:
                return self.prior_kernel_f_space
        else:
            if noise:
                return self.posterior_kernel_space
            else:
                return self.posterior_kernel_f_space

    def cholesky(self, prior=False, noise=False):
        if prior:
            if noise:
                return self.prior_cholesky_space
            else:
                return self.prior_cholesky_f_space
        else:
            if noise:
                return self.posterior_cholesky_space
            else:
                return self.posterior_cholesky_f_space

    def _check_process(self):
        self.location.check_dims(self.th_inputs)
        self.kernel.check_dims(self.th_inputs)
        self.mapping.check_dims(self.th_inputs)

        self.location.check_hypers(self.name + '_')
        self.kernel.check_hypers(self.name + '_')
        self.mapping.check_hypers(self.name + '_')

        self.location.check_potential()
        self.kernel.check_potential()
        self.mapping.check_potential()

        if self.degree is not None:
            self.degree.check_dims(None)
            self.degree.check_hypers(self.name + '_')
            self.degree.check_potential()

    def plot_mapping(self, params=None, domain=None, inputs=None, outputs=None, neval=100, title=None, label='mapping'):
        if params is None:
            params = self.get_params_current()
        if outputs is None:
            outputs = self.th_outputs
        if domain is None:
            domain = np.linspace(outputs.mean() - 2 * np.sqrt(outputs.var()),
                                 outputs.mean() + 2 * np.sqrt(outputs.var()), neval)
        #inv_transform = self.compiles['mapping_th'](domain, **params)
        #plt.plot(inv_transform, domain, label='mapping_th')

        #if domain is None:
        #    domain = np.linspace(outputs.mean() - 2*np.sqrt(outputs.var()), outputs.mean() + 2*np.sqrt(outputs.var()), neval)
        transform = self.compiles['mapping_inv_th'](domain, **params)
        plt.plot(domain, transform, label=label)

        if title is None:
            title = 'Mapping'
        plot_text(title, 'Domain y', 'Domain T(y)')

    def plot_kernel(self, params=None, space=None, inputs=None, centers=[1/10, 1/2, 9/10]):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        if inputs is None:
            inputs = self.inputs_values
        ksi = self.compiles['kernel_space_inputs'](space, inputs, **params).T
        for ind in centers:
            plt.plot(self.th_order, ksi[int(len(ksi)*ind), :], label='k(x_'+str(int(len(ksi)*ind))+')')
        plot_text('Kernel', 'Space x', 'Kernel value k(x,v)')

    def plot_concentration(self, params=None, space=None, color=True, figsize=(6, 6)):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        concentration_matrix = self.compiles['kernel_space'](space, **params)
        if color:
            plt.figure(None, figsize)
            v = np.max(np.abs(concentration_matrix))
            plt.imshow(concentration_matrix, cmap=cm.seismic, vmax=v, vmin=-v)
        else:
            plt.matshow(concentration_matrix)
        plot_text('Concentration', 'Space x', 'Space x', legend=False)

    def plot_location(self, params=None, space=None):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        plt.plot(self.th_order, self.compiles['location_space'](space, **params), label='location')
        plot_text('Location', 'Space x', 'Location value m(x)')


class CopulaProcess(StochasticProcess):
    def __init__(self, copula: StochasticProcess=None, marginal: Mapping=None):
        pass

import types

import numpy as np
import theano as th
from matplotlib import pyplot as plt, cm
from theano import tensor as tt
from theano.tensor import slinalg as tsl, nlinalg as tnl

from .hypers import Freedom
from .hypers.kernels import Kernel, KernelSum, KernelNoise
from .hypers.means import Mean
from .hypers.mappings import Mapping, Identity
from .stochastic import zero32, StochasticProcess
from ..libs.tensors import tt_to_cov, cholesky_robust, tt_to_bounded, tt_to_num
from ..libs.plots import plot_text, show, grid2d, plot_2d


class EllipticalProcess(StochasticProcess):
    def __init__(self, space=None, location: Mean=None, kernel: Kernel=None, mapping: Mapping=Identity(), degree: Freedom=None,
                 noisy=True, *args, **kwargs):
        #print('EllipticalProcess__init__')

        self.f_location = location
        self.f_degree = degree
        self.f_mapping = mapping
        if noisy:
            self.f_kernel = kernel
            self.f_kernel_noise = KernelSum(self.f_kernel, KernelNoise(name='Noise')) #self.th_space,
        else:
            self.f_kernel = kernel
            self.f_kernel_noise = self.f_kernel
        kwargs['space'] = space
        super().__init__(*args, **kwargs)

    def _check_hypers(self):

        self.f_location.check_dims(self.inputs)
        self.f_kernel_noise.check_dims(self.inputs)
        self.f_mapping.check_dims(self.inputs)

        self.f_location.check_hypers(self.name + '_')
        self.f_kernel_noise.check_hypers(self.name + '_')
        self.f_mapping.check_hypers(self.name + '_')

        self.f_location.check_potential()
        self.f_kernel_noise.check_potential()
        self.f_mapping.check_potential()

        if self.f_degree is not None:
            self.f_degree.check_dims(None)
            self.f_degree.check_hypers(self.name + '_')
            self.f_degree.check_potential()

    def default_hypers(self):
        x = self.inputs
        y = self.outputs
        return {**self.f_location.default_hypers_dims(x, y), **self.f_kernel_noise.default_hypers_dims(x, y),
                **self.f_mapping.default_hypers_dims(x, y)}

    def th_define_process(self):
        #print('stochastic_define_process')
        # Basic Tensors
        self.mapping_outputs = tt_to_num(self.f_mapping.inv(self.th_outputs))
        self.mapping_latent = tt_to_num(self.f_mapping(self.th_outputs))
        #self.mapping_scalar = tt_to_num(self.f_mapping.inv(self.th_scalar))

        self.prior_location_space = self.f_location(self.th_space)
        self.prior_location_inputs = self.f_location(self.th_inputs)

        self.prior_kernel_space = tt_to_cov(self.f_kernel_noise.cov(self.th_space))
        self.prior_kernel_inputs = tt_to_cov(self.f_kernel_noise.cov(self.th_inputs))
        self.prior_cholesky_space = cholesky_robust(self.prior_kernel_space)

        self.prior_kernel_f_space = self.f_kernel.cov(self.th_space)
        self.prior_kernel_f_inputs = self.f_kernel.cov(self.th_inputs)
        self.prior_cholesky_f_space = cholesky_robust(self.prior_kernel_f_space)

        self.cross_kernel_space_inputs = tt_to_num(self.f_kernel_noise.cov(self.th_space, self.th_inputs))
        self.cross_kernel_f_space_inputs = tt_to_num(self.f_kernel.cov(self.th_space, self.th_inputs))

        self.posterior_location_space = self.prior_location_space + self.cross_kernel_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.mapping_outputs - self.prior_location_inputs))
        self.posterior_location_f_space = self.prior_location_space + self.cross_kernel_f_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.mapping_outputs - self.prior_location_inputs))

        self.posterior_kernel_space = self.prior_kernel_space - self.cross_kernel_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.cross_kernel_space_inputs.T))
        self.posterior_cholesky_space = cholesky_robust(self.posterior_kernel_space)

        self.posterior_kernel_f_space = self.prior_kernel_f_space - self.cross_kernel_f_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.cross_kernel_f_space_inputs.T))
        self.posterior_cholesky_f_space = cholesky_robust(self.posterior_kernel_f_space)

    def th_freedom(self, prior=False, noise=False):
        if prior:
            return self.f_degree()
        else:
            return self.f_degree() + self.th_inputs.shape[0].astype(th.config.floatX)

    def th_mapping_inv(self, prior=False, noise=False):
        return self.mapping_outputs

    def th_mapping(self, prior=False, noise=False):
        return self.mapping_latent

    def th_location(self, prior=False, noise=False):
        if prior:
            return self.prior_location_space
        else:
            if noise:
                return self.posterior_location_space
            else:
                return self.posterior_location_f_space

    def th_kernel(self, prior=False, noise=False):
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

    def th_cholesky(self, prior=False, noise=False):
        return cholesky_robust(self.th_kernel(prior=prior, noise=noise))

    def th_kernel_diag(self, prior=False, noise=False):
        return tt_to_bounded(tnl.extract_diag(self.th_kernel(prior=prior, noise=noise)), zero32)

    def th_kernel_sd(self, prior=False, noise=False, cholesky=False):
        r = self.th_kernel_diag(prior=prior, noise=noise)
        if cholesky:
            return cholesky_robust(tt.nlinalg.alloc_diag(r))
        else:
            return tt.sqrt(r)

    def th_median(self, prior=False, noise=False):
        debug_p('median' + str(prior) + str(noise))
        return self.f_mapping(self.th_location(prior=prior, noise=noise))

    def th_mean(self, prior=False, noise=False, simulations=None):
        debug_p('mean' + str(prior) + str(noise))
        return self.f_mapping(self.th_location(prior=prior, noise=noise))

    def th_variance(self, prior=False, noise=False):
        debug_p('variance' + str(prior) + str(noise))
        return self.th_kernel_diag(prior=prior, noise=noise)

    def th_covariance(self, prior=False, noise=False):
        debug_p('covariance' + str(prior) + str(noise))
        return self.th_kernel(prior=prior, noise=noise)

    def _compile_methods(self):
        super()._compile_methods()
        self.freedom = types.MethodType(self._method_name('th_freedom'), self)
        self.mapping = types.MethodType(self._method_name('th_mapping'), self)
        self.mapping_inv = types.MethodType(self._method_name('th_mapping_inv'), self)
        self.location = types.MethodType(self._method_name('th_location'), self)
        self.kernel = types.MethodType(self._method_name('th_kernel'), self)
        self.cholesky = types.MethodType(self._method_name('th_cholesky'), self)
        self.kernel_diag = types.MethodType(self._method_name('th_kernel_diag'), self)
        self.kernel_sd = types.MethodType(self._method_name('th_kernel_sd'), self)
        self.cross_mean = types.MethodType(self._method_name('th_cross_mean'), self)

    def plot_kernel(self, params=None, space=None, inputs=None, prior=True, noise=False, centers=[1/10, 1/2, 9/10]):
        if params is None:
            params = self.params
        if space is None:
            space = self.space
        if inputs is None:
            inputs = self.inputs
        ksi = self.kernel(params=params, space=space, inputs=inputs, prior=prior, noise=noise).T
        for ind in centers:
            plt.plot(self.order, ksi[int(len(ksi)*ind), :], label='k(x_'+str(int(len(ksi)*ind))+')')
        plot_text('Kernel', 'Space x', 'Kernel value k(x,v)')

    def plot_concentration(self, params=None, space=None, prior=True, noise=True, color=True, cmap=cm.seismic, figsize=(6, 6), title='Concentration'):
        if params is None:
            params = self.params
        if space is None:
            space = self.space
        concentration_matrix = self.kernel(params=params, space=space, prior=prior, noise=noise)
        if color:
            if figsize is not None:
                plt.figure(None, figsize)
            v = np.max(np.abs(concentration_matrix))
            plt.imshow(concentration_matrix, cmap=cmap, vmax=v, vmin=-v)
        else:
            plt.matshow(concentration_matrix)
        plot_text(title, 'Space x', 'Space x', legend=False)

    def plot_mapping(self, params=None, domain=None, inputs=None, outputs=None, neval=100, title=None, label='mapping'):
        if params is None:
            params = self.params
        if domain is None:
            if outputs is None:
                outputs = self.outputs
            mean = np.mean(outputs)
            std = np.std(outputs)
            domain = np.linspace(mean - 2 * std, mean + 2 * std, neval)
            domain = np.linspace(outputs.min(), outputs.max(), neval)
        #plt.plot(self.mapping(params=params, outputs=domain, inputs=inputs), domain, label=label)
        plt.plot(domain, self.mapping_inv(params=params, outputs=domain, inputs=inputs), label=label)

        if title is None:
            title = 'Mapping'
        plot_text(title, 'Domain y', 'Domain T(y)')

    def plot_model(self, params=None, indexs=None, kernel=True, mapping=True, marginals=True, bivariate=True):
        if indexs is None:
            indexs = [self.index[len(self.index)//2], self.index[len(self.index)//2]+1]
        if kernel:
            plt.subplot(121)
            self.plot_kernel(params=params)
        if mapping:
            plt.subplot(122)
            self.plot_mapping(params=params)
        show()

        if marginals:
            plt.subplot(121)
            self.plot_distribution(index=indexs[0], params=params, space=self.space[indexs[0]:indexs[0]+1, :], prior=True)
            self.plot_distribution(index=indexs[0], params=params, space=self.space[indexs[0]:indexs[0]+1, :])
            plt.subplot(122)
            self.plot_distribution(index=indexs[1], params=params, space=self.space[indexs[1]:indexs[1]+1, :], prior=True)
            self.plot_distribution(index=indexs[1], params=params, space=self.space[indexs[1]:indexs[1]+1, :])
            show()
        if bivariate:
            self.plot_distribution2D(indexs=indexs, params=params, space=self.space[indexs, :])
            show()

    def plot_distribution(self, index=0, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, quantiles_noise=False, noise=False, prior=False, sigma=4, neval=100, title=None, swap=False, label=None):
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, quantiles_noise=quantiles_noise, noise=noise, distribution=True, prior=prior)
        domain = np.linspace(pred.mean - sigma * pred.std, pred.mean + sigma * pred.std, neval)
        dist_plot = np.zeros(len(domain))
        for i in range(len(domain)):
            dist_plot[i] = pred.logpredictive(domain[i:i + 1])
        dist_plot = np.exp(dist_plot)
        if label is None:
            if prior:
                label='prior'
            else:
                label='posterior'
        if label is False:
            label = None
        if title is None:
            title = 'Marginal Distribution y_'+str(self.order[index])
        if swap:
            plt.plot(dist_plot, domain, label=label)
            plot_text(title, 'Density', 'Domain y')
        else:
            plt.plot(domain, dist_plot, label=label)
            plot_text(title, 'Domain y', 'Density')

    def plot_distribution2D(self, indexs=[0, 1], params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, quantiles_noise=False, noise=False, prior=False, sigma_1=2, sigma_2=2, neval=33, title=None):
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, quantiles_noise=quantiles_noise, noise=noise, distribution=True, prior=prior)
        dist1 = np.linspace(pred.mean[0] - sigma_1 * pred.std[0], pred.mean[0] + sigma_1 * pred.std[0], neval)
        dist2 = np.linspace(pred.mean[1] - sigma_2 * pred.std[1], pred.mean[1] + sigma_2 * pred.std[1], neval)
        xy, x2d, y2d = grid2d(dist1, dist2)
        dist_plot = np.zeros(len(xy))
        for i in range(len(xy)):
            dist_plot[i] = pred.logpredictive(xy[i])
        dist_plot = np.exp(dist_plot)
        plot_2d(dist_plot, x2d, y2d)
        if title is None:
            title = 'Distribution2D'
        plot_text(title, 'Domain y_'+str(self.order[indexs[0]]), 'Domain y_'+str(self.order[indexs[1]]), legend=False)

    # TODO: Check
    def plot_kernel2D(self):
        pass

    def plot_location(self, params=None, space=None):
        if params is None:
            params = self.params
        if space is None:
            space = self.th_space
        plt.plot(self.order, self.compiles.location_space(space, **params), label='location')
        plot_text('Location', 'Space x', 'Location value m(x)')


def debug_p(*args, **kwargs):
    pass#print(*args, **kwargs)#pass#

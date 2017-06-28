import pymc3 as pm


def modelcontext(model=None):
    return pm.modelcontext(model)

def trans_hypers(hypers):
    trans = DictObj()
    for k, v in hypers.items():
        if type(k) is pm.model.TransformedRV:
            trans[k.transformed] = k.transformed.distribution.transform_used.forward(v).eval()
        else:
            trans[k] = v
    return trans

def def_space(space=None, name=None, squeeze=False):
    if space is None:
        space = np.arange(0, 2, dtype=th.config.floatX)
        space = space[:, None]
    elif np.isscalar(space):
        space_x = np.arange(0, space, dtype=th.config.floatX)
        space_x = space_x[None, :]
        return th.shared(space_x, name, borrow=True), None, None

    if squeeze:
        space = np.squeeze(space)
        if type(space) is np.ndarray:
            space_x = space.astype(th.config.floatX)
            if name is None:
                space_th = None
            else:
                space_th = th.shared(space_x, name, borrow=True)
            if len(space_x.shape) == 1 or space_x.shape[1] == 1:
                space_t = np.squeeze(space_x)
            else:
                space_t = np.arange(len(space_x), dtype=th.config.floatX)
        else:
            space_t = space.index
            space_x = space.astype(th.config.floatX)
            if name is None:
                space_th = None
            else:
                space_th = th.shared(space_x.values, name, borrow=True)
        return space_th, space_x, space_t

    if type(space) is np.ndarray:
        if len(space.shape) < 2:
            space = space[:, None]
        space_x = space.astype(th.config.floatX)
        if name is None:
            space_th = None
        else:
            space_th = th.shared(space_x, name, borrow=True)
        if len(space_x.shape) == 1 or space_x.shape[1] == 1:
            space_t = np.squeeze(space_x)
        else:
            space_t = np.arange(len(space_x), dtype=th.config.floatX)
    else:
        space_t = space.index
        space_x = space.astype(th.config.floatX)
        if len(space.shape) < 2:
            space_x = pd.DataFrame(space_x)
        if name is None:
            space_th = None
        else:
            space_th = th.shared(space_x.values, name, borrow=True)
    return space_th, space_x, space_t.astype(np.int32)



    @staticmethod
    def Null(name, shape=(), testval=zeros):
        with modelcontext():
            return pm.NoDistribution(name, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def Flat(name, shape=(), testval=zeros):
        with modelcontext():
            return pm.Flat(name, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def ExpFlat(name, shape=(), testval=ones):
        with modelcontext():
            return pm.Flat(name, transform=pm.distributions.transforms.log, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def FlatExp(name, shape=(), testval=ones):
        with modelcontext():
            return pm.Flat(name, transform=non_transform_log, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def FlatPos(name, shape=(), testval=ones):
        with modelcontext():
            return PositiveFlat(name, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def FlatExpId(name, shape=(), testval=ones):
        with modelcontext():
            return pm.Flat(name, transform=LogIdTransform(), shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def Exponential(name, lam=ones, shape=(), testval=ones):
        with modelcontext():
            return pm.Exponential(name, shape=shape, lam=lam(shape), testval=testval(shape), dtype=th.config.floatX)


class _StochasticProcess:
    """Abstract class used to define a StochasticProcess.

    Attributes:
        model (pm.Model): Reference to the context pm.Model
    """
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, mapping: Mapping=None, noise=True, freedom=None,
                 name=None, inputs=None, outputs=None, hidden=None, description=None, file=None, recompile=False, precompile=False,
                 multioutput=False):


        # Space, Hidden, Observed
        if type(space) is int:
            space = np.random.rand(2, space).astype(dtype=th.config.floatX)
        space_raw = space
        space = space_raw[:2]
        __, self.space_values, self.space_index = def_space(space)
        self.inputs, self.inputs_values, self.observed_index = def_space(space, self.name + '_inputs')
        self.outputs, self.outputs_values, __ = def_space(np.zeros(len(space)), self.name + '_outputs', squeeze=True)


        self.random_th = tt.vector(self.name + '_random_th', dtype=th.config.floatX)
        self.random_scalar = tt.scalar(self.name + '_random_scalar', dtype=th.config.floatX)


        self.random_th.tag.test_value = np.random.randn(len(self.space_values)).astype(dtype=th.config.floatX)
        self.random_scalar.tag.test_value = np.float32(10)

        if hidden is None:
            self.hidden = None
        else:
            self.hidden = np.squeeze(hidden)



        print('Space Dimension: ', self.space_values.shape[1])

        # Hyper-parameters values
        self._widget_samples = 0
        self._widget_traces = None
        self.params_current = None
        self.params_widget = None
        self.params_fixed = DictObj()
        self._fixed_keys = []
        self._fixed_array = None
        self._fixed_chain = None
        self.sampling_dims = None
        self.fixed_dims = None





        self.compiles = DictObj()
        self.compiles_trans = DictObj()
        self.compiles_transf = DictObj()
        print('Init Definition')
        self._define_process()
        print('Definition OK')

        self._compile(precompile)
        self._compile_transforms(precompile)
        print('Compilation OK')

        self.observed(inputs, outputs)

        self.logp_prior = None

        if not multioutput:
            self._compile_logprior()
            #__, self.space_values, self.space_index = def_space(space_raw)
            self.fix_params()

        self.set_space(space_raw, self.hidden)

    def _compile_logprior(self):
        self.logp_prior = self.model.bijection.mapf(self.model.fn(tt.add(*map(tt.sum, [var.logpt for var in self.model.free_RVs] + self.model.potentials))))

    def logp_array(self, params):
        return self.model.logp_array(params)

    def dlogp_array(self, params):
        return self.model.dlogp_array(params)

    def dict_to_array(self, params):
        return self.model.dict_to_array(params)

    def array_to_dict(self, params):
        return self.model.bijection.rmap(params)

    def logp_dict(self, params):
        return self.model.logp_array(self.model.dict_to_array(params))

    def logp_chain(self, chain):
        out = np.empty(len(chain))
        for i in range(len(out)):
            out[i] = self.model.logp_array(chain[i])
        return out

    def logp_fixed(self, params):
        if len(params) > len(self.sampling_dims):
            params = params[self.sampling_dims]
        self._fixed_array[self.sampling_dims] = params
        return self.model.logp_array(self._fixed_array)

    #@jit
    def _logp_fixed(self, sampling_params):
        self._fixed_array[self.sampling_dims] = sampling_params
        return self.model.logp_array(self._fixed_array)

    def _logp_fixed_prior(self, sampling_params):
        self._fixed_array[self.sampling_dims] = sampling_params
        return self.logp_prior(self._fixed_array)

    def _logp_fixed_like(self, sampling_params):
        self._fixed_array[self.sampling_dims] = sampling_params
        return self.model.logp_array(self._fixed_array) - self.logp_prior(self._fixed_array)

    @jit
    def _dlogp_fixed(self, sampling_params):
        self._fixed_array[self.sampling_dims] = sampling_params
        return self.model.dlogp_array(self._fixed_array)[self.sampling_dims]

    @jit
    def logp_fixed_chain(self, params):
        if len(params) > len(self.sampling_dims):
            params = params[self.sampling_dims]
        return self._logp_fixed_chain(params)

    @jit
    def _logp_fixed_chain(self, sampling_params):
        self._fixed_chain[:, self.sampling_dims] = sampling_params
        out = np.empty(len(self._fixed_chain))
        for i in range(len(out)):
            out[i] = self.model.logp_array(self._fixed_chain[i])
        return out.mean()

    @jit
    def _dlogp_fixed_chain(self, sampling_params):
        self._fixed_chain[:, self.sampling_dims] = sampling_params
        out = np.empty((len(self._fixed_chain), len(self.sampling_dims)))
        for i in range(len(out)):
            out[i, :] = self.model.dlogp_array(self._fixed_chain[i])[self.sampling_dims]
        return out.mean(axis=0)

    def fix_params(self, fixed_params=None):
        if fixed_params is None:
            fixed_params = DictObj()
        self.params_fixed = fixed_params
        self._fixed_keys = list(self.params_fixed.keys())
        self._fixed_array = self.dict_to_array(self.get_params_default())
        self._fixed_chain = None
        self.calc_dimensions()

    def _compile(self, precompile=False):
        params = [self.space_th] + self.model.vars
        self.compiles['location_space'] = makefn(params, self.location_space, precompile)
        self.compiles['kernel_space'] = makefn(params, self.kernel_space, precompile)
        self.compiles['kernel_f_space'] = makefn(params, self.kernel_f_space, precompile)

        params = [self.inputs_th] + self.model.vars
        self.compiles['location_inputs'] = makefn(params, self.location_inputs, precompile)
        self.compiles['kernel_inputs'] = makefn(params, self.kernel_inputs, precompile)
        self.compiles['kernel_f_inputs'] = makefn(params, self.kernel_f_inputs, precompile)

        params = [self.space_th, self.inputs_th] + self.model.vars
        self.compiles['kernel_space_inputs'] = makefn(params, self.kernel_space_inputs, precompile)
        self.compiles['kernel_f_space_inputs'] = makefn(params, self.kernel_f_space_inputs, precompile)

        params = self.model.vars
        self.compiles['mapping_outputs'] = makefn(params, self.mapping_outputs, precompile)
        self.compiles['mapping_th'] = makefn([self.random_th] + params, self.mapping_th, precompile)
        self.compiles['mapping_inv_th'] = makefn([self.random_th] + params, self.mapping_inv_th, precompile)

        params = [self.space_th] + self.model.vars
        self.compiles['prior_mean'] = makefn(params, self.prior_mean, precompile)
        self.compiles['prior_covariance'] = makefn(params, self.prior_covariance, precompile)
        self.compiles['prior_cholesky'] = makefn(params, self.prior_cholesky, precompile)
        self.compiles['prior_variance'] = makefn(params, self.prior_variance, precompile)
        self.compiles['prior_std'] = makefn(params, self.prior_std, precompile)
        self.compiles['prior_noise'] = makefn(params, self.prior_noise, precompile)
        self.compiles['prior_median'] = makefn(params, self.prior_median, precompile)
        self.compiles['prior_quantile_up'] = makefn(params, self.prior_quantile_up, precompile)
        self.compiles['prior_quantile_down'] = makefn(params, self.prior_quantile_down, precompile)
        self.compiles['prior_noise_up'] = makefn(params, self.prior_noise_up, precompile)
        self.compiles['prior_noise_down'] = makefn(params, self.prior_noise_down, precompile)
        self.compiles['prior_logp'] = makefn([self.random_th] + params, self.prior_logp, precompile)
        self.compiles['prior_logpred'] = makefn([self.random_th] + params, self.prior_logpred, precompile)
        self.compiles['prior_distribution'] = makefn([self.random_th] + params, self.prior_distribution, precompile)
        try:
            self.compiles['prior_sampler'] = makefn([self.random_th] + params, self.prior_sampler, precompile)
        except:
            self.compiles['prior_sampler'] = makefn([self.random_scalar, self.random_th] + params, self.prior_sampler, precompile)

        params = [self.space_th, self.inputs_th, self.outputs_th] + self.model.vars
        self.compiles['posterior_mean'] = makefn(params, self.posterior_mean, precompile)
        self.compiles['posterior_covariance'] = makefn(params, self.posterior_covariance, precompile)
        self.compiles['posterior_cholesky'] = makefn(params, self.posterior_cholesky, precompile)
        self.compiles['posterior_variance'] = makefn(params, self.posterior_variance, precompile)
        self.compiles['posterior_std'] = makefn(params, self.posterior_std, precompile)
        self.compiles['posterior_noise'] = makefn(params, self.posterior_noise, precompile)
        self.compiles['posterior_median'] = makefn(params, self.posterior_median, precompile)
        self.compiles['posterior_quantile_up'] = makefn(params, self.posterior_quantile_up, precompile)
        self.compiles['posterior_quantile_down'] = makefn(params, self.posterior_quantile_down, precompile)
        self.compiles['posterior_noise_up'] = makefn(params, self.posterior_noise_up, precompile)
        self.compiles['posterior_noise_down'] = makefn(params, self.posterior_noise_down, precompile)
        self.compiles['posterior_logp'] = makefn([self.random_th] + params, self.posterior_logp, precompile)
        self.compiles['posterior_logpred'] = makefn([self.random_th] + params, self.posterior_logpred, precompile)
        self.compiles['posterior_distribution'] = makefn([self.random_th] + params, self.posterior_distribution, precompile)
        try:
            self.compiles['posterior_sampler'] = makefn([self.random_th] + params, self.posterior_sampler, precompile)
        except:
            self.compiles['posterior_sampler'] = makefn([self.random_scalar, self.random_th] + params, self.posterior_sampler, precompile)

    def _compile_transforms(self, precompile=False):
        params = [self.random_th]
        for v in self.model.vars:
            dist = v.distribution
            if hasattr(dist, 'transform_used'):
                self.compiles_trans[str(v)] = makefn(params, dist.transform_used.backward(self.random_th), precompile)
                self.compiles_transf[str(v)] = makefn(params, dist.transform_used.forward(self.random_th), precompile)

    def set_space(self, space, hidden=None, index=None):
        __, self.space_values, self.space_index = def_space(space)
        self.hidden = hidden
        if index is not None:
            self.space_index = index

    def observed(self, inputs=None, outputs=None, index=None):
        if inputs is None or outputs is None or len(inputs) == 0 or len(inputs) == 0:
            self.inputs_values, self.outputs_values, self.observed_index = None, None, index
            self.inputs.set_value(self.inputs_values, borrow=True)
            self.outputs.set_value(self.outputs_values, borrow=True)
            return

        __, self.inputs_values, self.observed_index = def_space(inputs)
        if index is not None:
            self.observed_index = index
        __, self.outputs_values, __ = def_space(outputs, squeeze=True)
        self.inputs.set_value(self.inputs_values, borrow=True)
        self.outputs.set_value(self.outputs_values, borrow=True)
        if self.distribution is None:
            with self.model:
                self._define_distribution()

    def prior(self, params=None, space=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, samples=0, distribution=False):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        values = DictObj()
        if mean:
            values['mean'] = self.compiles['prior_mean'](space, **params)
        if var:
            values['variance'] = self.compiles['prior_variance'](space, **params)
            values['std'] = self.compiles['prior_std'](space, **params)
        if cov:
            values['covariance'] = self.compiles['prior_covariance'](space, **params)
        if median:
            values['median'] = self.compiles['prior_median'](space, **params)
        if quantiles:
            values['quantile_up'] = self.compiles['prior_quantile_up'](space, **params)
            values['quantile_down'] = self.compiles['prior_quantile_down'](space, **params)
        if noise:
            values['noise'] = self.compiles['prior_noise'](space, **params)
            values['noise_up'] = self.compiles['prior_noise_up'](space, **params)
            values['noise_down'] = self.compiles['prior_noise_down'](space, **params)
        # TODO: if samples is with another distribution
        if samples > 0:
            S = np.empty((len(space), samples))
            rand = np.random.randn(len(space), samples)
            for i in range(samples):
                S[:, i] = self.compiles['prior_sampler'](rand[:, i], space, **params)
                values['samples'] = S
        if distribution:
            values['logp'] = lambda x: self.compiles['prior_logp'](x, space, **params)
            values['logpred'] = lambda x: self.compiles['prior_logpred'](x, space, **params)
            values['distribution'] = lambda x: self.compiles['prior_distribution'](x, space, **params)
        return values

    def posterior(self, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, samples=0, distribution=False):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        if inputs is None:
            inputs = self.inputs_values
        if outputs is None:
            outputs = self.outputs_values
        values = DictObj()
        if mean:
            values['mean'] = self.compiles['posterior_mean'](space, inputs, outputs, **params)
        if var:
            values['variance'] = self.compiles['posterior_variance'](space, inputs, outputs, **params)
            values['std'] = self.compiles['posterior_std'](space, inputs, outputs, **params)
        if cov:
            values['covariance'] = self.compiles['posterior_covariance'](space, inputs, outputs, **params)
        if median:
            values['median'] = self.compiles['posterior_median'](space, inputs, outputs, **params)
        if quantiles:
            values['quantile_up'] = self.compiles['posterior_quantile_up'](space, inputs, outputs, **params)
            values['quantile_down'] = self.compiles['posterior_quantile_down'](space, inputs, outputs, **params)
        if noise:
            values['noise'] = self.compiles['posterior_noise'](space, inputs, outputs, **params)
            values['noise_up'] = self.compiles['posterior_noise_up'](space, inputs, outputs, **params)
            values['noise_down'] = self.compiles['posterior_noise_down'](space, inputs, outputs, **params)
        # TODO: if samples is with another distribution
        if samples > 0:
            S = np.empty((len(space), samples))
            rand = np.random.randn(len(space), samples)
            for i in range(samples):
                S[:, i] = self.compiles['posterior_sampler'](rand[:, i], space, inputs, outputs, **params)
            values['samples'] = S
        if distribution:
            values['logp'] = lambda x: self.compiles['posterior_logp'](x, space, inputs, outputs, **params)
            values['logpred'] = lambda x: self.compiles['posterior_logpred'](x, space, inputs, outputs, **params)
            values['distribution'] = lambda x: self.compiles['posterior_distribution'](x, space, inputs, outputs, **params)
        return values

    def sample(self, params=None, space=None, inputs=None, outputs=None, samples=1, prior=False):
        S = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=False, var=False, cov=False, median=False, quantiles=False, noise=False, samples=samples, prior=False)
        return S['samples']

    def predict(self, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, samples=0, distribution=False, prior=False):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        if inputs is None:
            inputs = self.inputs_values
        if outputs is None:
            outputs = self.outputs_values
        if prior or self.observed_index is None:
            return self.prior(params=params, space=space, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, samples=samples, distribution=distribution)
        else:
            return self.posterior(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, samples=samples, distribution=distribution)

    def scores(self, params=None, space=None, hidden=None, inputs=None, outputs=None, logp=True, bias=True, variance=False, median=False):
        if hidden is None:
            hidden = self.hidden
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, var=variance, median=median, distribution=logp)
        scores = DictObj()
        if bias:
            scores['_BiasL1'] = np.mean(np.abs(pred.mean - hidden))
            scores['_BiasL2'] = np.mean((pred.mean - hidden)**2)
        if variance:
            scores['_MSE'] = np.mean((pred.mean - hidden) ** 2 + pred.variance)
            scores['_RMSE'] = np.sqrt(scores['_MSE'])
        if median:
            scores['_MedianL1'] = np.mean(np.abs(pred.median - hidden))
            scores['_MedianL2'] = np.mean((pred.median - hidden)**2)
        if logp:
            #scores['_NLL'] = - pred.logp(hidden) / len(hidden)
            scores['_NLPD'] = - pred.logpred(hidden) / len(hidden)
        return scores




class GaussianProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, noise=True,
                 name='GP', inputs=None, outputs=None, hidden=None, file=None, precompile=False, *args, **kwargs):
        super().__init__(space=space, location=location, kernel=kernel, mapping=Identity(), noise=noise,
                         freedom=None, name=name, inputs=inputs, outputs=outputs, hidden=hidden, file=file, precompile=precompile, *args, **kwargs)

    def _define_process(self):
        # Prior

        self.prior_logp = TGPDist.logp_cho(self.random_th, self.prior_mean, self.prior_cholesky, self.mapping)
        self.prior_distribution = tt.exp(self.prior_logp.sum())
        self.prior_logpred = TGPDist.logp_cho(self.random_th, self.prior_mean, nL.alloc_diag(self.prior_noise),
                                              self.mapping)
        # Posterior
        self.posterior_logp = TGPDist.logp_cho(self.random_th, self.posterior_mean, self.posterior_cholesky, self.mapping)
        self.posterior_distribution = tt.exp(self.posterior_logp.sum())
        self.posterior_logpred = TGPDist.logp_cho(self.random_th, self.posterior_mean, nL.alloc_diag(self.posterior_noise), self.mapping)

    def subprocess(self, subkernel):
        k_cross = subkernel.cov(self.space_th, self.inputs_th)
        subprocess_mean = self.location_space + k_cross.dot(tsl.solve(self.kernel_inputs, self.mapping_outputs - self.location_inputs))
        params = [self.space_th, self.inputs_th, self.outputs_th] + self.model.vars
        return makefn(params, subprocess_mean, True)


class TransformedGaussianProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, mapping: Mapping=None, noise=True,
                 name='TGP', inputs=None, outputs=None, hidden=None, file=None, precompile=False, *args, **kwargs):
        super().__init__(space=space, location=location, kernel=kernel, mapping=mapping, noise=noise,
                         freedom=None, name=name, inputs=inputs, outputs=outputs, hidden=hidden,  file=file, precompile=precompile, *args, **kwargs)

    def _define_process(self, n=10):

        # Latent
        self.prior_logp = TGPDist.logp_cho(self.random_th, self.latent_prior_mean, self.prior_cholesky, self.mapping)
        self.prior_logpred = TGPDist.logp_cho(self.random_th, self.latent_prior_mean, nL.alloc_diag(self.latent_prior_noise), self.mapping)
        self.prior_distribution = tt.exp(self.prior_logp.sum())

        self.posterior_logp = TGPDist.logp_cho(self.random_th, self.latent_posterior_mean, self.posterior_cholesky, self.mapping)
        self.posterior_logpred = TGPDist.logp_cho(self.random_th, self.latent_posterior_mean, nL.alloc_diag(self.latent_posterior_noise), self.mapping)
        self.posterior_distribution = tt.exp(self.posterior_logp.sum())
        print('Latent OK')
        self.prior_covariance = None
        self.posterior_covariance = None


import pickle
import numpy as np
import pymc3 as pm
import scipy as sp
import theano as th
import theano.tensor as tt
import theano.tensor.nlinalg as nL
import theano.tensor.slinalg as sL
from pymc3.distributions.distribution import generate_samples
from scipy.stats._multivariate import multivariate_normal

from .libs.tensors import cholesky_robust, tt_to_num, debug


class Model(pm.Model):

    @classmethod
    def get_contexts(cls):
        return pm.Model.get_contexts()

    @classmethod
    def get_context(cls):
        try:
            return pm.Model.get_context()
        except TypeError:
            with pm.Model() as model:
                return model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior = False

    def use_prior(self, prior=True):
        self.prior = prior

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

    def dlogp(self, vars=None):
        """Nan Robust dlogp"""
        return self.model.fn(tt_to_num(pm.gradient(self.logpt, vars)))

    def fastdlogp(self, vars=None):
        """Nan Robust fastdlogp"""
        return self.model.fastfn(tt_to_num(pm.gradient(self.logpt, vars)))




def load_model(path):
    with Model():
        with open(path, 'rb') as f:
            try:
                r = pickle.load(f)
                print('Loaded model ' + path)
                return r
            except:
                print('Error loading model '+path)


class TGPDist(pm.Continuous):
    def __init__(self, mu, cov, mapping, tgp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov
        self.mapping = mapping
        self.tgp = tgp

    def logp_cov(self, value):  # es más rápido pero se cae
        delta = tt_to_num(self.mapping.inv(value)) - self.mu
        return -np.float32(0.5) * (tt.log(nL.det(self.cov)) + delta.T.dot(sL.solve(self.cov, delta))
                                   + self.cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))) \
               + self.mapping.logdet_dinv(value)

    def logp_cho(self, value):
        delta = tt_to_num(self.mapping.inv(value)) - self.mu
        L = sL.solve_lower_triangular(self.cho, delta)
        return -np.float32(0.5) * (self.cov.shape[0].astype(th.config.floatX) * tt.log(np.float32(2.0 * np.pi))
                                   + L.T.dot(L)) - tt.sum(tt.log(nL.diag(self.cho))) + self.mapping.logdet_dinv(value)

    def logp(self, value):
        if False:
            return tt_to_num(debug(self.logp_cov(value), 'logp_cov'), -np.inf, -np.inf)
        else:
            return tt_to_num(debug(self.logp_cho(value), 'logp_cho'), -np.inf, -np.inf)

    @property
    def cho(self):
        try:
            return cholesky_robust(self.cov)
        except:
            raise sp.linalg.LinAlgError("not cholesky")

    def random(self, point=None, size=None):
        point = self.tgp.point(self.tgp, point)
        mu = self.tgp.compiles['mean'](**point)
        cov = self.tgp.compiles['covariance'](**point)

        def _random(mean, cov, size=None):
            return self.compiles['trans'](multivariate_normal.rvs(mean, cov, None if size == mean.shape else size), **point)
        samples = generate_samples(_random, mean=mu, cov=cov, dist_shape=self.shape, broadcast_shape=mu.shape, size=size)
        return samples


class ConstantStep(pm.step_methods.arraystep.ArrayStep):
    """
    Dummy sampler that returns the current value at every iteration. Useful for
    fixing parameters at a particular value.

    Parameters
    ----------
    vars : list
    List of variables for sampler.
    model : PyMC Model
    Optional model for sampling step. Defaults to None (taken from context).
    """

    def __init__(self, vars, model=None, **kwargs):
        model = pm.modelcontext(model)
        self.model = model
        vars = pm.inputvars(vars)
        super(ConstantStep, self).__init__(vars, [model.fastlogp], **kwargs)

    def astep(self, q0, logp):
        return q0


class RobustSlice(pm.step_methods.arraystep.ArrayStep):
    """
    Univariate slice sampler step method

    Parameters
    ----------
    vars : list
        List of variables for sampler.
    w : float
        Initial width of slice (Defaults to 1).
    tune : bool
        Flag for tuning (Defaults to True).
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    max_iter: int
        Max Iterations before to return the same point (Defaults to 25)

    """
    default_blocked = False

    def __init__(self, vars=None, w=1., tune=True, model=None, max_iter=25, **kwargs):
        self.model = pm.modelcontext(model)
        self.w = w
        self.tune = tune
        self.w_sum = 0
        self.n_tunes = 0
        self.max_iter = max_iter
        if vars is None:
            vars = self.model.cont_vars
        vars = pm.inputvars(vars)

        super(RobustSlice, self).__init__(vars, [self.model.fastlogp], **kwargs)

    def astep(self, q0, logp):
        self.w = np.resize(self.w, len(q0))
        y = logp(q0) - np.random.standard_exponential()

        # Stepping out procedure
        q_left = q0 - np.random.uniform(0, self.w)
        q_right = q_left + self.w


        #print('q0, log(q0), y=logp(q0)-N(0,1)',q0, logp(q0), y)
        #print('w, q_left, q_right',self.w, q_left, q_right)
        #print('y, log(q_left), log(q_right)',y, logp(q_left), logp(q_right))

        iter = 0
        while (y < logp(q_left)).all() and iter<self.max_iter:
            q_left -= self.w
            iter+=1

        while (y < logp(q_right)).all() and iter<self.max_iter:
            q_right += self.w
            iter += 1
        if iter >= self.max_iter:
            return q0

        q = np.random.uniform(q_left, q_right, size=q_left.size)  # new variable to avoid copies
        while logp(q) <= y:
            # Sample uniformly from slice
            if (q > q0).all():
                q_right = q
            elif (q < q0).all():
                q_left = q
            q = np.random.uniform(q_left, q_right, size=q_left.size)

        if self.tune:
            # Tune sampler parameters
            self.w_sum += np.abs(q0 - q)
            self.n_tunes += 1.
            self.w = 2. * self.w_sum / self.n_tunes
        return q

    @staticmethod
    def competence(var):
        if var.dtype in pm.continuous_types:
            if not var.shape:
                return pm.Competence.PREFERRED
            return pm.Competence.COMPATIBLE
        return pm.Competence.INCOMPATIBLE
import pickle
import numpy as np
import pymc3 as pm
import scipy as sp
import theano as th
import theano.tensor as tt

from pymc3.distributions.distribution import generate_samples
from scipy.stats._multivariate import multivariate_normal

from .libs.tensors import cholesky_robust, tt_to_num, debug



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
import pickle
import numpy as np
import pymc3 as pm
import scipy as sp
import theano as th
import theano.tensor as tt

from pymc3.distributions.distribution import generate_samples
from scipy.stats._multivariate import multivariate_normal

from .libs.tensors import cholesky_robust, tt_to_num, debug
from pymc3 import modelcontext, inputvars, Point, discrete_types
from pymc3.tuning.scaling import find_hessian_diag, fixed_hessian,  adjust_scaling
from pymc3.step_methods.hmc import unif, quad_potential, SamplerHist,\
    HamiltonianMC, Hamiltonian, leapfrog, energy, metrop_select, Competence


def guess_scaling(point, vars=None, model=None, scaling_bound=1e-8):
    model = modelcontext(model)
    try:
        h = find_hessian_diag(point, vars, model=model)
    except:
        h = fixed_hessian(point, vars, model=model)
    return adjust_scaling(h, scaling_bound)


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

    def __init__(self, vars=None, w=1., tune=True, model=None, max_iter=10, **kwargs):
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

        iter = 0
        while (y < logp(q_left)).all() and iter < self.max_iter:
            q_left -= self.w
            iter += 1

        while (y < logp(q_right)).all() and iter < self.max_iter:
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


class RobustHamiltonianMC(pm.step_methods.arraystep.ArrayStep):
    default_blocked = True

    def __init__(self, vars=None, scaling=None, step_scale=.25, path_length=2., is_cov=False, step_rand=unif, state=None, model=None, **kwargs):
        """
        Parameters
        ----------
            vars : list of theano variables
            scaling : array_like, ndim = {1,2}
                Scaling for momentum distribution. 1d arrays interpreted matrix diagonal.
            step_scale : float, default=.25
                Size of steps to take, automatically scaled down by 1/n**(1/4) (defaults to .25)
            path_length : float, default=2
                total length to travel
            is_cov : bool, default=False
                Treat scaling as a covariance matrix/vector if True, else treat it as a precision matrix/vector
            step_rand : function float -> float, default=unif
                A function which takes the step size and returns an new one used to randomize the step size at each iteration.
            state
                State object
            model : Model
        """
        model = modelcontext(model)

        if vars is None:
            vars = model.cont_vars
        vars = inputvars(vars)

        if scaling is None:
            scaling = model.test_point

        if isinstance(scaling, dict):
            scaling = guess_scaling(Point(scaling, model=model), model=model)

        n = scaling.shape[0]

        self.step_size = step_scale / n ** (1 / 4.)

        self.potential = quad_potential(scaling, is_cov, as_cov=False)

        self.path_length = path_length
        self.step_rand = step_rand

        if state is None:
            state = SamplerHist()
        self.state = state

        super(RobustHamiltonianMC, self).__init__(
            vars, [model.fastlogp, model.fastdlogp(vars)], **kwargs)

    def astep(self, q0, logp, dlogp):
        H = Hamiltonian(logp, dlogp, self.potential)

        e = self.step_rand(self.step_size)
        nstep = int(self.path_length / e)

        p0 = H.pot.random()

        q, p = leapfrog(H, q0, p0, nstep, e)
        p = -p

        mr = energy(H, q0, p0) - energy(H, q, p)

        self.state.metrops.append(mr)

        return metrop_select(mr, q, q0)

    @staticmethod
    def competence(var):
        if var.dtype in discrete_types:
            return Competence.INCOMPATIBLE
        return Competence.COMPATIBLE

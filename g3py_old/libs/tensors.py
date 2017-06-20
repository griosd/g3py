import copy
import numpy as np
import scipy as sp
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as sT
from IPython.display import Image
from pymc3.memoize import memoize
from theano.compile.ops import as_op

th.config.on_unused_input = 'ignore'


def debug(x, name='', force=False):
    if th.config.mode in ['NanGuardMode', 'DebugMode'] or force:
        return th.printing.Print(name)(x)
    else:
        return x


class makefn:
    def __init__(self, th_vars, fn, precompile=False):
        self.th_vars = th_vars
        self.fn = fn
        if precompile:
            self.compiled = th.function(self.th_vars, self.fn, allow_input_downcast=True, on_unused_input='ignore')
        else:
            self.compiled = None

    def __call__(self, *args, **kwargs):
        if self.compiled is None:
            self.compiled = th.function(self.th_vars, self.fn, allow_input_downcast=True, on_unused_input='ignore')
        return self.compiled(*args, **kwargs)

#@memoize
#def makefn(th_vars, fn):
#    return th.function(th_vars, fn, allow_input_downcast=True, on_unused_input='ignore')


def show_graph(f, name='temp.png'):
    th.printing.pydotprint(f, with_ids=True, outfile=name, var_with_name_simple=True)
    return Image(name)


def print_graph(f):
    return th.printing.debugprint(f)


def inf_to_num(r, neg=-1e20, pos=1e20):
    return tt.switch(tt.isinf(r) and r > 0, np.float32(pos), tt.switch(tt.isinf(r), np.float32(neg), r))


def tt_to_num(r, nan=0, inf=1e10):
    return tt.switch(tt.isnan(r), np.float32(nan), tt.switch(tt.isinf(r), np.nan_to_num(np.float32(inf)), r))


def tt_to_cov(c):
    r = tt_to_num(c)
    m = tt.min(tt.diag(r))
    return tt.switch(m > 0, r, r + (1e-6-m)*tt.eye(c.shape[0]) )


def inverse_function(func, z, tol=1e-3, n_steps=1024, alpha=0.1):
    def iter_newton(x):
        diff = (func(x) - z)
        dfunc = tt.grad(tt.sum(diff), x)
        dfunc = tt.switch(tt.abs_(dfunc) < 1.0, tt.sgn(dfunc), dfunc)
        return x - alpha*diff/dfunc, th.scan_module.until(tt.max(tt.abs_(diff)) < tol)
    init = 0*z
    values, _ = th.scan(iter_newton, outputs_info=init, n_steps=n_steps)
    return values[-1]


class InverseFunction(th.gof.Op):
    def __init__(self, func):
        self.func = func

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        return th.gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        try:
            z[0] = sp.optimize.root(lambda _x: self.func(_x) - x, x).astype(x.dtype)
        except:
            raise

    def grad(self, inputs, gradients):
        pass
        x = inputs[0]
        dz = gradients[0]
        inv_x = self(x)


class CholeskyRobust(th.gof.Op):
    """
    Return a triangular matrix square root of positive semi-definite `x`.

    L = cholesky(X, lower=True) implies dot(L, L.T) == X.

    """

    __props__ = ('lower', 'destructive')

    def __init__(self):
        self.lower = True
        self.destructive = False
        self.maxtries = 20

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        assert x.ndim == 2
        return th.gof.Apply(self, [x], [x.type()])

    def _cholesky(self, K):
        L, info = sp.linalg.lapack.dpotrf(K, lower=1)
        if info == 0:
            return L
        # else:
        #print(K)
        diagK = np.diag(K)
        dK = np.eye(K.shape[0]) * diagK.mean() * 1e-6
        if np.any(diagK <= 0.0):
            K = K + np.eye(K.shape[0]) * (diagK.mean() * 1e-6 - diagK.min() )
            #raise sp.linalg.LinAlgError("not positive-definite: negative diagonal element")
        for num_tries in range(self.maxtries):
            try:
                return np.nan_to_num(sp.linalg.cholesky(K + dK, lower=True))
            except:
                dK *= 10
        raise sp.linalg.LinAlgError("not approximate positive-definite")

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        try:
            z[0] = self._cholesky(x).astype(x.dtype)
        except:
            z[0] = (0*x + 1e-10*np.eye(len(x))).astype(x.dtype)
            #raise sp.linalg.LinAlgError("not perform cholesky")

    def grad(self, inputs, gradients):
        """
        Cholesky decomposition reverse-mode gradient update.

        Symbolic expression for reverse-mode Cholesky gradient taken from [0]_

        References
        ----------
        .. [0] I. Murray, "Differentiation of the Cholesky decomposition",
           http://arxiv.org/abs/1602.07527

        """
        x = inputs[0]
        dz = gradients[0]
        chol_x = self(x)

        # deal with upper triangular by converting to lower triangular
        if not self.lower:
            chol_x = chol_x.T
            dz = dz.T

        def tril_and_halve_diagonal(mtx):
            """Extracts lower triangle of square matrix and halves diagonal."""
            return tt.tril(mtx) - tt.diag(tt.diagonal(mtx) / 2.)

        def conjugate_solve_triangular(outer, inner):
            """Computes L^{-T} P L^{-1} for lower-triangular L."""
            return solve_upper_triangular(
                outer.T, solve_upper_triangular(outer.T, tt_to_num(inner).T).T)

        s = conjugate_solve_triangular(
            chol_x, tril_and_halve_diagonal(tt_to_num(chol_x.T.dot(dz))))

        if self.lower:
            return [tt.tril(s + s.T) - tt.diag(tt.diagonal(s))]
        else:
            return [tt.triu(s + s.T) - tt.diag(tt.diagonal(s))]


cholesky_robust = CholeskyRobust()

try:
    solve_lower_triangular = sT.solve_lower_triangular
    solve_upper_triangular = sT.solve_upper_triangular
except:
    solve_lower_triangular = sT.Solve(A_structure='lower_triangular', lower=True)
    solve_upper_triangular = sT.Solve(A_structure='upper_triangular', lower=False)
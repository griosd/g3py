import numpy as np
import scipy as sp
import pymc3 as pm
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as tsl
from IPython.display import Image
from . import clone


def gradient1(f, v):
    """flat gradient of f wrt v"""
    return tt.flatten(tt.grad(f, v, disconnected_inputs='warn'))


def gradient(f, wrt=None):
    if wrt is None:
        wrt = pm.inputvars(pm.cont_inputs(f))
    if wrt:
        return tt.concatenate([gradient1(f, v) for v in wrt], axis=0)
    else:
        return tt.zeros(0.0, dtype='float32')


def debug(x, name='', force=False):
    if th.config.mode in ['NanGuardMode', 'DebugMode'] or force:
        try:
            return th.printing.Print(name)(x)
        except Exception as m:
            print(name, m)
            return x
    else:
        return x

class makefn:
    def __init__(self, th_vars, fn, givens=None, bijection=None, precompile=False):
        self.th_vars = th_vars
        self.fn = fn
        self.givens = givens
        self.bijection = bijection
        if precompile:
            #print(self.th_vars, self.fn)
            self.compiled = th.function(self.th_vars, self.fn, givens=self.givens, allow_input_downcast=True,
                                        on_unused_input='ignore')
        else:
            self.compiled = None
        self.executed = 0

    def __call__(self, params, space=None, inputs=None, outputs=None, vector=[]):
        self.executed += 1
        if self.compiled is None:
            #print(self.th_vars, self.fn)
            self.compiled = th.function(self.th_vars, self.fn, givens=self.givens, allow_input_downcast=True,
                                        on_unused_input='ignore')
        if self.givens is None:
            if self.bijection is None:
                return self.compiled( **params)
            else:
                return self.compiled(**self.bijection(params))
        elif vector is None:
            if self.bijection is None:
                return self.compiled(space, inputs, outputs, **params)
            else:
                return self.compiled(space, inputs, outputs, **self.bijection(params))
        else:
            if self.bijection is None:
                return self.compiled(space, inputs, outputs, vector, **params)
            else:
                return self.compiled(space, inputs, outputs, vector, **self.bijection(params))

    def clone(self, bijection=None):
        r = clone(self)
        r.bijection = bijection
        return r


def show_graph(f, name='temp.png'):
    th.printing.pydotprint(f, with_ids=True, outfile=name, var_with_name_simple=True)
    return Image(name)


def print_graph(f):
    return th.printing.debugprint(f)


def inf_to_num(r, neg=-np.float32(1e20), pos=np.float32(1e20)):
    return tt.switch(tt.isinf(r) and r > np.float32(0), np.float32(pos), tt.switch(tt.isinf(r), np.float32(neg), r))


def tt_to_num(r, nan=np.float32(0), inf=np.float32(1e10)):
    return tt.switch(tt.isnan(r), np.float32(nan), tt.switch(tt.isinf(r), np.nan_to_num(np.float32(inf)), r))


def tt_to_cov(c):
    r = tt_to_num(c)
    m = tt.min(tt.diag(r))
    return tt.switch(m > np.float32(0), r, r + (np.float32(1e-6)-m)*tt.eye(c.shape[0]) )


def tt_to_bounded(r, lower=None, upper=None):
    if lower is None and upper is None:
        return r
    if lower is None:
        return tt.switch(r > upper, upper, r)
    if upper is None:
        return tt.switch(r < lower, lower, r)
    return tt.switch(r < lower, lower, tt.switch(r > upper, upper, r))


class EvalOp(th.gof.Op):

    view_map = {0: [0]}

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return th.gof.Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def c_code_cache_version(self):
        return (1,)

tt_eval = EvalOp()


def inverse_function(func, z, tol=1e-3, n_steps=1024, alpha=0.1):
    def iter_newton(x, z1):
        diff = (func(x) - z1)
        dfunc = tt.grad(tt.sum(diff), x)
        dfunc = tt.switch(tt.abs_(dfunc) < 1.0, tt.sgn(dfunc), dfunc)
        r = x - alpha*diff/dfunc
        return r, th.scan_module.until(tt.max(tt.abs_(diff)) < tol)
    init = 0*z
    values, _ = th.scan(iter_newton, outputs_info=init, non_sequences=z, n_steps=n_steps)
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
        dK = np.eye(K.shape[0]) * diagK.mean() * np.float32(1e-6)
        if np.any(diagK <= 0.0):
            K = K + np.eye(K.shape[0]) * (diagK.mean() * np.float32(1e-6) - diagK.min() )
            #raise sp.linalg.LinAlgError("not positive-definite: negative diagonal element")
        for num_tries in range(self.maxtries):
            try:
                return np.nan_to_num(sp.linalg.cholesky(K + dK, lower=True))
            except:
                dK *= np.float32(10)
        raise sp.linalg.LinAlgError("not approximate positive-definite")

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        try:
            z[0] = self._cholesky(x).astype(x.dtype)
        except:
            z[0] = (0*x + np.float32(1e-10)*np.eye(len(x))).astype(x.dtype)
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
    solve_lower_triangular = tsl.solve_lower_triangular
    solve_upper_triangular = tsl.solve_upper_triangular
except:
    solve_lower_triangular = tsl.Solve(A_structure='lower_triangular', lower=True)
    solve_upper_triangular = tsl.Solve(A_structure='upper_triangular', lower=False)
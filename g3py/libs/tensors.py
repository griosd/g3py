import numpy as np
import scipy as sp
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as sT
from IPython.display import Image
th.config.on_unused_input = 'ignore'



def debug(x, name=''):
    if th.config.mode in ['NanGuardMode', 'DebugMode']:
        return th.printing.Print(name)(x)
    else:
        return x


def makefn(vars, fn):
    return th.function(vars, fn, allow_input_downcast=True, on_unused_input='ignore')


def show_graph(f, name='temp.png'):
    th.printing.pydotprint(f, with_ids=True, outfile=name, var_with_name_simple=True)
    return Image(name)


def tt_to_num(r, nan=0, inf=np.inf):
    return tt.switch(tt.isnan(r), np.nan_to_num(np.float32(nan)), tt.switch(tt.isinf(r), np.nan_to_num(np.float32(inf)), r))


def tt_to_cov(c):
    r = tt_to_num(c)
    m = tt.min(tt.diag(r))
    return tt.switch(m > 0, r, r + (1e-6-m)*tt.eye(c.shape[0]) )


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
        # L, info = sp.linalg.lapack.dpotrf(K, lower=1)
        # if info == 0:
        #    return L
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
            raise sp.linalg.LinAlgError("not perform cholesky")

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
                outer.T, solve_upper_triangular(outer.T, inner.T).T)

        s = conjugate_solve_triangular(
            chol_x, tril_and_halve_diagonal(tt_to_num(chol_x.T.dot(dz))))

        if self.lower:
            return [tt.tril(s + s.T) - tt.diag(tt.diagonal(s))]
        else:
            return [tt.triu(s + s.T) - tt.diag(tt.diagonal(s))]


cholesky_robust = CholeskyRobust()
solve_lower_triangular = sT.solve_lower_triangular
solve_upper_triangular = sT.solve_upper_triangular

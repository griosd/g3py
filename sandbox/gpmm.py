import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sb


def plot_style():
    sb.set(style="white", color_codes=True) #white background
    plt.rcParams['figure.figsize'] = (20, 6)  #figure size
    plt.rcParams['axes.titlesize'] = 20  # title size
    plt.rcParams['axes.labelsize'] = 18  # xy-label size
    plt.rcParams['xtick.labelsize'] = 16 #x-numbers size
    plt.rcParams['ytick.labelsize'] = 16 #y-numbers size
    plt.rcParams['legend.fontsize'] = 18  # legend size
    #plt.rcParams['legend.fancybox'] = True
    pass
plot_style()


def plot(title="title", x="xlabel", y="ylabel", loc=0):
    plt.axis('tight')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    if not loc is None:
        plt.legend(loc=loc)


# Kernels
class KernelFunction:
    def __init__(self, _noise, _theta):
        self.noise = _noise
        self.theta = _theta

    def setParameters(self, _param):
        self.noise = _param[0]
        self.theta = _param[1:]

    def dot(self, X, _x):
        return X.dot(_x)

    def sample(self, X):
        return CovCholesky(self, X).dot(np.random.randn(len(X), 1))

    def params(self):
        return np.concatenate([np.array([self.noise]), self.theta])

    def cov(self, X1, X2=None):
        if X2 is None:
            cov = np.empty((len(X1), len(X1)))
            i = 0
            for row in X1:
                TempK = self.dot(X1[i:], row)
                cov[i, i:] = TempK
                cov[i + 1:, i] = TempK[1:].T
                i += 1
            return cov + np.exp(self.noise) * np.eye(len(X1))
        else:
            n1 = len(X1)
            n2 = len(X2)
            cov = np.empty((n1, n2))
            i = 0
            if (n1 < n2):
                for row in X1:
                    cov[i, :] = self.dot(X2, row)
                    i += 1
            else:
                for row in X2:
                    cov[:, i] = self.dot(X1, row)
                    i += 1
            return cov


def CovCholesky(K, X=None, maxtries=5):
    if X is not None:
        K = K.cov(X)
    # A = np.ascontiguousarray(A)
    L, info = sp.linalg.lapack.dpotrf(K, lower=1)
    if info == 0:
        return L
    else:
        dK = np.diag(K)
        if np.any(dK <= 0.0):
            raise sp.linalg.LinAlgError("not positive-definite: negative diagonal element")
        dK = np.eye(K.shape[0]) * dK.mean() * 1e-6
        for num_tries in range(maxtries):
            try:
                L = sp.linalg.cholesky(K + dK, lower=True)
                return L
            except:
                dK *= 10
        raise sp.linalg.LinAlgError("not approximate positive-definite")
    return L


def SolveCho(Lx, Fx):
    return sp.linalg.solve_triangular(Lx, Fx, lower=True)


def Solve2Cho(Lx, Fx):
    try:
        return sp.linalg.cho_solve((Lx, True), Fx)
    except:
        return sp.linalg.cho_solve((np.nan_to_num(Lx), True), Fx)


def MeanPredFromCho(Kyx, Lx, Fx):
    return Kyx.dot(Solve2Cho(Lx, Fx))


def VarPredFromCho(Kyx, Lx, Kyy):
    Ly_x = SolveCho(Lx, Kyx.T)
    return np.diag(Kyy) - np.diag(Ly_x.T.dot(Ly_x))


def CovPredFromCho(Kyx, Lx, Kyy):
    Ly_x = SolveCho(Lx, Kyx.T)
    return Kyy - Ly_x.T.dot(Ly_x)


def NLLFromCholAndObs(Lx, Fx):
    const = 0.5 * len(Fx) * np.log(2 * np.pi)
    det = np.sum(np.log(np.diag(Lx)))
    dist = 0.5 * Fx.T.dot(Solve2Cho(Lx, Fx))
    return const + det + dist


def NLL(params, Kernel, X, Fx):
    Kernel.setParameters(params)
    return NLLFromCholAndObs(CovCholesky(Kernel, X), Fx)


def LearningGP(Kernel, X, Fx, _method='BFGS'):
    return sp.optimize.minimize(NLL, Kernel.params(), args=(Kernel, X, Fx), method=_method, options={'disp': True})


class PredictorGP:
    def __init__(self, kernel, X, Fx):
        self.kernel = kernel
        self.X = X
        self.Fx = Fx
        self.Lx = CovCholesky(self.kernel.cov(self.X))

    def __call__(self, Y, Plot=True, Desv=True, nS=2):
        Kyy = self.kernel.cov(Y, Y)
        Kyx = self.kernel.cov(Y, self.X)
        Fy_x = MeanPredFromCho(Kyx, self.Lx, self.Fx)
        DKy_x = VarPredFromCho(Kyx, self.Lx, Kyy)
        if Plot:
            for i in range(len(self.X[0])):
                plt.figure(i + 1, figsize=(20, 6))
                plt.plot(self.X[:, i], np.squeeze(self.Fx), '.r', ms=10,label='observations')
                plt.plot(Y[:, i], np.squeeze(Fy_x), '--k', linewidth=2, label='mean')
                if Desv:
                    mean = np.squeeze(Fy_x)
                    std = np.sqrt(DKy_x)
                    plt.fill_between(Y[:, i], mean + nS * std, mean - nS * std, alpha=0.2, label='95%')
        return (Fy_x, DKy_x)

    def sample(self, Y, n=1, Plot=True):
        if n < 1:
            return None
        Kyy = self.kernel.cov(Y, Y)
        Kyx = self.kernel.cov(Y, self.X)
        Fy_x = MeanPredFromCho(Kyx, self.Lx, self.Fx)
        Cy_x = CovPredFromCho(Kyx, self.Lx, Kyy)
        Ly_x = CovCholesky(Cy_x)
        rand = np.random.randn(len(Y), n)
        sample = Fy_x[:, np.newaxis] + Ly_x.dot(rand)
        if Plot:
            for i in range(len(self.X[0])):
                plt.figure(i + 1, figsize=(20, 6))
                plt.plot(Y[:, i], sample)
                plt.plot(self.X[:, i], np.squeeze(self.Fx), '.r', ms=10)
        return sample



class KernelMM(KernelFunction):
    def __init__(self, _kern, _steps, _noise, _w):
        self.kern = _kern
        self.steps = _steps
        self.noise = _noise
        self.w = _w
        self.limw = len(self.w) + 1

    def setParameters(self, _param):
        self.noise = _param[0]
        self.w = _param[1:self.limw]
        self.kern.setParameters(_param[self.limw:])

    def params(self):
        return np.concatenate([np.array([self.noise]), self.w, self.kern.params()])

    def dot(self, X, _x):
        return self.cov(X,_x)
        w = (self.w + self.w[::-1]) ** 2
        w = w / np.sum(w)
        Xsteps = X + self.steps
        _xsteps = _x + self.steps
        C = self.kern.cov(Xsteps.reshape((Xsteps.size, 1)), _xsteps.reshape((_xsteps.size, 1)))
        return np.kron(np.eye(X.size), w).dot(C).dot(np.kron(np.eye(_x.size), w).T)

    def cov(self, X, X2=None):
        w = (self.w + self.w[::-1]) ** 2
        w = w / np.sum(w)
        xsteps = (X[:, np.newaxis] + self.steps).reshape((len(X) * len(self.steps), X.shape[1]))
        if X2 is None:
            full_cov = self.kern.cov(xsteps)
            full_cov = np.kron(np.eye(len(X)), w).dot(full_cov).dot(np.kron(np.eye(len(X)), w).T) + np.exp(
                self.noise) * np.eye(len(X))
        else:
            x2steps = (X2[:, np.newaxis] + self.steps).reshape((len(X2) * len(self.steps), X2.shape[1]))
            full_cov = self.kern.cov(xsteps, x2steps)
            full_cov = np.kron(np.eye(len(X)), w).dot(full_cov).dot(np.kron(np.eye(len(X2)), w).T)
        return full_cov

    def cross(self, X, _x):
        w = (self.w + self.w[::-1]) ** 2
        w = w / np.sum(w)
        Xsteps = X
        _xsteps = (_x[:, np.newaxis] + self.steps).reshape((len(_x) * len(self.steps), _x.shape[1]))
        Cw = self.kern.cov(Xsteps, _xsteps)
        return Cw.dot(np.kron(np.eye(len(_x)), w).T)


class PredictorCrossGP:
    def __init__(self, kernel, X, Fx):
        self.kernel = kernel
        self.X = X
        self.Fx = Fx
        self.Lx = CovCholesky(self.kernel.cov(self.X))

    def __call__(self, Y, Plot=True, Desv=True, nS=2):
        Kyy = self.kernel.cov(Y, Y)
        Kyx = self.kernel.cov(Y, self.X)
        Fy_x = MeanPredFromCho(Kyx, self.Lx, self.Fx)
        DKy_x = VarPredFromCho(Kyx, self.Lx, Kyy)
        if Plot:
            for i in range(len(self.X[0])):
                plt.figure(i + 1, figsize=(20, 6))
                plt.plot(self.X[:, i], np.squeeze(self.Fx), '.r', ms=10)
                plt.plot(Y[:, i], np.squeeze(Fy_x), '--k', linewidth=2)
                if Desv:
                    mean = np.squeeze(Fy_x)
                    std = np.sqrt(DKy_x)
                    plt.fill_between(Y[:, i], mean + nS * std, mean - nS * std, alpha=0.2, label='95%')
        return (Fy_x, DKy_x)

    def sample(self, Y, n=1, Plot=True):
        Kyy = self.kernel.cov(Y, Y)
        Kyx = self.kernel.cov(Y, self.X)
        Fy_x = MeanPredFromCho(Kyx, self.Lx, self.Fx)
        Cy_x = CovPredFromCho(Kyx, self.Lx, Kyy)

        Ly_x = CovCholesky(Cy_x)
        rand = np.random.randn(len(Y), n)
        sample = Fy_x[:, np.newaxis] + Ly_x.dot(rand)

        # sample = np.random.multivariate_normal(Fy_x, Cy_x, n).T

        if Plot:
            for i in range(len(self.X[0])):
                plt.figure(i + 1, figsize=(20, 6))
                plt.plot(Y[:, i], sample)
                plt.plot(self.X[:, i], np.squeeze(self.Fx), '.r', ms=10)

        return sample

    def cross(self, Y):
        Kyy = self.kernel.kern.cov(Y, Y)
        Kyx = self.kernel.cross(Y, self.X)
        Fy_x = MeanPredFromCho(Kyx, self.Lx, self.Fx)

        DKy_x = VarPredFromCho(Kyx, self.Lx, Kyy)

        return (Fy_x, DKy_x)


class PositiveConstKernel(KernelFunction):
    def __init__(self, _noise=1, _c=0):
        self.noise = _noise
        self.theta = np.array([_c])

    def dot(self, X, _x):
        return np.ones(len(X)) * np.exp(self.theta[0])


class SimpleKernelFunction(KernelFunction):
    def __init__(self, _noise=1, _H=1, _W=np.array([1])):
        self.noise = _noise
        self.theta = np.concatenate([np.array([_H]), _W])


class GaussianKernel(SimpleKernelFunction):
    def dot(self, X, _x):
        return np.exp(self.theta[0]) * np.exp(-((X - _x) ** 2).dot(np.exp(self.theta[1:])))


class LaplaceKernel(SimpleKernelFunction):
    def dot(self, X, _x):
        return np.exp(self.theta[0]) * np.exp(-np.abs(X - _x).dot(np.exp(self.theta[1:])))


class LinearKernel(KernelFunction):
    def __init__(self, _noise=1, _C=1, _L=np.array([1]), _W=np.array([1])):
        self.noise = _noise
        self.theta = np.concatenate([np.array([_C]), _L, _W])

    def dot(self, X, _x):
        s = np.exp(self.theta[0])
        L = self.theta[1:(len(self.theta) - 1) // 2 + 1]
        S = np.exp(self.theta[(len(self.theta) - 1) // 2 + 1:])
        return np.sum(s + S * (X - L) * (_x - L), 1)


class NeuralNetKernel(KernelFunction):
    def __init__(self, _noise=1, _H=1, _C=1, _L=np.array([1]), _W=np.array([1])):
        self.noise = _noise
        self.theta = np.concatenate([np.array([_H, _C]), _L, _W])

    def dot(self, X, _x):
        h = np.exp(self.theta[0])
        s = np.exp(self.theta[1])
        L = self.theta[2:(len(self.theta) - 2) // 2 + 2]
        S = np.exp(self.theta[(len(self.theta) - 2) // 2 + 2:])
        X = X - L
        _x = _x - L
        Xdot_x = 2 * np.sum((s + S * X * _x), 1) / np.sqrt(
            (1 + 2 * np.sum(s + S * (X ** 2), 1)) * (1 + 2 * np.sum(s + S * (_x ** 2))))
        return h * np.arcsin(Xdot_x)


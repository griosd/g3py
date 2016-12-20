import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from g3py.sandbox.gpmm import CovCholesky, MeanPredFromCho, CovPredFromCho, VarPredFromCho, NLLFromCholAndObs


class TransBoxCox:
    def __init__(self, _power=1, _shift=0, _scale=0, _learn=True):
        self.power = _power
        self.shift = _shift
        self.scale = _scale
        self.learn = _learn

    def setParameters(self, _power, _shift, _scale):
        if self.learn:
            self.power = _power
            self.shift = _shift
            #self.scale = _scale

    def __call__(self, X):
        p_exp = np.abs(self.power)
        scale = np.exp(self.scale)
        scaled_X = scale * X + self.shift
        if p_exp == 0:
            return np.log(scaled_X)
        else:
            return (np.sign(scaled_X)*np.power(np.abs(scaled_X), p_exp)-1)/p_exp

    def inv(self,Y):
        p_exp = np.abs(self.power)
        scale = np.exp(self.scale)
        if p_exp == 0:
            return (np.exp(Y)-self.shift)/scale
        else:
            p_expY = p_exp*Y+1
            trans_Y = np.sign(p_expY) * (np.abs(p_expY)) ** (1 / p_exp)
            return (trans_Y-self.shift)/scale

    def NLL(self, X):
        p_exp = np.abs(self.power)
        scale = np.exp(self.scale)
        scaled_X = np.abs(scale * X + self.shift)
        if p_exp == 0:
            return np.sum(np.log(scaled_X)) - len(X)*np.log(scale)
        else:
            return (1 - p_exp)*np.sum(np.log(scaled_X)) - len(X)*np.log(scale)

    def __len__(self):
        return 3

    @property
    def params(self):
        return np.array([self.power, self.shift, self.scale])



class TShiftBoxCox:
    def __init__(self, _power=1, _shift=0, _learn=True):
        self.power = _power
        self.shift = _shift
        self.learn = _learn

    def setParameters(self,_power,_shift):
        if self.learn:
            self.power = _power
            self.shift = _shift

    def __call__(self, X):
        p_exp = np.abs(self.power)
        scaled_X = X + self.shift
        if p_exp == 0:
            return np.log(scaled_X)
        else:
            return (np.sign(scaled_X)*np.power(np.abs(scaled_X), p_exp)-1)/p_exp

    def inv(self,Y):
        p_exp = np.abs(self.power)
        if p_exp == 0:
            return np.exp(Y)-self.shift
        else:
            p_expY = p_exp*Y+1
            trans_Y = np.sign(p_expY) * (np.abs(p_expY)) ** (1 / p_exp)
            return trans_Y-self.shift

    def NLL(self, X):
        p_exp = np.abs(self.power)
        scaled_X = np.abs(X + self.shift)
        if p_exp == 0:
            return np.sum(np.log(scaled_X))
        else:
            return (1 - p_exp)*np.sum(np.log(scaled_X))

    def __len__(self):
        return 2

    @property
    def params(self):
        return np.array([self.power, self.shift])


class ShiftLogT:
    def __init__(self, _shift=1):
        self.shift = _shift

    def setParameters(self, _shift):
        self.shift = _shift

    def __call__(self, X):
        return np.sign(X + self.shift) * np.log(np.abs(X + self.shift))  ## Revisar

    def inv(self, Y):
        return np.exp(Y) - self.shift

    def NLL(self, X):
        return np.sum(np.sign(X + self.shift) * np.log(np.abs(X + self.shift)))

    def __len__(self):
        return 1

    @property
    def params(self):
        return np.array([self.shift])


class LambdaPowerT:
    def __init__(self, _power=1):
        self.power = _power

    def setParameters(self, _power):
        self.power = _power

    def __call__(self, X):
        return np.sign(X) * np.power(np.abs(X), self.power)  ## Revisar

    def inv(self, Y):
        return np.sign(Y) * np.power(np.abs(Y), 1 / self.power)

    def NLL(self, X):
        return -np.sum(
            np.sign(self.power) * np.log(np.abs(self.power)) + np.sign(X) * (self.power - 1) * np.log(np.abs(X)))

    def __len__(self):
        return 1

    @property
    def params(self):
        return np.array([self.power])


def TNLL(params, Kernel, T, X, Fx):
    n = len(T)
    if n == 1:
        T.setParameters(params[0])
    elif n == 2:
        T.setParameters(params[0], params[1])
    elif n == 3:
        T.setParameters(params[0], params[1], params[2])
    Kernel.setParameters(params[n:])
    return NLLFromCholAndObs(CovCholesky(Kernel, X), T(Fx)) + T.NLL(Fx)


def LearningTGP(Kernel, T, X, Fx, _method='BFGS'):
    return sp.optimize.minimize(TNLL, np.concatenate([T.params, np.array([Kernel.noise]), Kernel.theta]),
                                args=(Kernel, T, X, Fx), method=_method, options={'disp': True})


def gauss_hermite(f, mu, sigma, n=100):
    a, w = np.polynomial.hermite.hermgauss(n)
    return w.dot(f(mu + np.sqrt(2)*sigma*a) )/np.sqrt(np.pi)

class PredictorTGP:
    def __init__(self, kernel, T, X, Fx):
        self.kernel = kernel
        self.T = T
        self.X = X
        self.Fx = Fx
        self.Lx = CovCholesky(self.kernel.cov(self.X))

    def __call__(self, Y, Plot=True, Desv=True, nS=2):
        Kyy = self.kernel.cov(Y, Y)
        Kyx = self.kernel.cov(Y, self.X)
        TFy_x = MeanPredFromCho(Kyx, self.Lx, self.T(self.Fx))
        DKy_x = VarPredFromCho(Kyx, self.Lx, Kyy)

        std = np.sqrt(DKy_x)
        mean = np.zeros(len(Y))
        var = np.zeros(len(Y))
        for i in range(len(Y)):
            mean[i] = gauss_hermite(self.T.inv, TFy_x[i], std[i], 10)
            var[i] = gauss_hermite(lambda v:self.T.inv(v)**2, TFy_x[i], std[i], 10) - mean[i]**2
        Fy_x = self.T.inv(TFy_x + 0.5 * DKy_x)  ##### OJO CON ESTO, SOLO ES VERDAD EN EL LOG
        if Plot:
            for i in range(len(self.X[0])):
                plt.figure(i + 1, figsize=(20, 6))
                plt.plot(self.X[:, i], np.squeeze(self.Fx), '.r', ms=10,label='observations')  ## observaciones
                plt.plot(Y[:, i], mean, 'k--', linewidth=2,label='mean')  ## media
                #plt.plot(Y[:, i], np.squeeze(self.T.inv(TFy_x)), 'g--', linewidth=1,label='median')  ## mediana
                if Desv:
                    #plt.plot(Y[:, i], np.squeeze(self.T.inv(TFy_x - std)), 'k--',
                    #         linewidth=1)  ##### OJO CON ESTO, SOLO ES VERDAD EN EL LOG
                    plt.fill_between(Y[:, i], self.T.inv(TFy_x + nS * std), self.T.inv(TFy_x - nS * std), alpha=0.2,
                                     label='95%')
                plt.plot(self.X[:, i], np.squeeze(self.Fx), '.r', ms=10)
        return (mean, var)

    def sample(self, Y, n=1, Plot=True):
        if n < 1:
            return None
        Kyy = self.kernel.cov(Y, Y)
        Kyx = self.kernel.cov(Y, self.X)
        TFy_x = MeanPredFromCho(Kyx, self.Lx, self.T(self.Fx))
        Cy_x = CovPredFromCho(Kyx, self.Lx, Kyy)
        Ly_x = CovCholesky(Cy_x)

        rand = np.random.randn(len(Y), n)
        sample = self.T.inv(TFy_x[:, np.newaxis] + Ly_x.dot(rand))

        if Plot:
            for i in range(len(self.X[0])):
                plt.figure(i + 1, figsize=(20, 6))
                plt.plot(Y[:, i], sample)
                plt.plot(self.X[:, i], np.squeeze(self.Fx), '.r', ms=10)
        return sample
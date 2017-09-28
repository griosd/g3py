from .stochastic import StochasticProcess
from .marginal import MarginalProcess


class CopulaProcess(StochasticProcess):
    def __init__(self, copula: StochasticProcess=None, marginal: MarginalProcess=None):
        self.copula = copula
        self.marginal = marginal


class CopulaGaussianProcess(CopulaProcess):
    pass


class CopulaStudentTProcess(CopulaProcess):
    pass


class TransformedStudentTProcess(CopulaProcess):
    pass

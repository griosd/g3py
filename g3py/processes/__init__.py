from .stochastic import StochasticProcess
from .gaussian import GaussianProcess, WarpedGaussianProcess
from .studentT import StudentTProcess, WarpedStudentTProcess
from .marginal import *
from .copula import CopulaGaussianProcess, CopulaStudentTProcess


GP = GaussianProcess
WGP = WarpedGaussianProcess

TP = StudentTProcess
WTP = WarpedStudentTProcess

CGP = CopulaGaussianProcess
CSTP = CopulaStudentTProcess

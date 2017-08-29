from .stochastic import StochasticProcess
from .gaussian import GaussianProcess, WarpedGaussianProcess
from .studentT import StudentTProcess
from .copula import CopulaGaussianProcess, CopulaStudentTProcess

GP = GaussianProcess
WGP = WarpedGaussianProcess
STP = StudentTProcess
CGP = CopulaGaussianProcess
CSTP = CopulaStudentTProcess

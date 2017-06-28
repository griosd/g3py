from .stochastic import StochasticProcess
from .gaussian import GaussianProcess, TransformedGaussianProcess
from .studentT import StudentTProcess
from .copula import CopulaGaussianProcess, CopulaStudentTProcess

GP = GaussianProcess
STP = StudentTProcess
TGP = TransformedGaussianProcess
TSTP = TransformedGaussianProcess
CGP = CopulaGaussianProcess
CSTP = CopulaStudentTProcess

from .stochastic import StochasticProcess
from .gaussian import GaussianProcess
from .studentT import StudentTProcess
from .copula import TransformedGaussianProcess, TransformedGaussianProcess, CopulaGaussianProcess, CopulaStudentTProcess

GP = GaussianProcess
STP = StudentTProcess
TGP = TransformedGaussianProcess
TSTP = TransformedGaussianProcess
CGP = CopulaGaussianProcess
CSTP = CopulaStudentTProcess

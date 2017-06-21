from .stochastic import EllipticalProcess, CopulaProcess
from .gaussian import GaussianProcess
from .studentT import StudentTProcess


class CopulaGaussianProcess(EllipticalProcess, CopulaProcess, GaussianProcess):
    pass


class CopulaStudentTProcess(EllipticalProcess, CopulaProcess, StudentTProcess):
    pass


class TransformedGaussianProcess(EllipticalProcess, CopulaProcess, GaussianProcess):
    pass


class TransformedStudentTProcess(EllipticalProcess, CopulaProcess, StudentTProcess):
    pass

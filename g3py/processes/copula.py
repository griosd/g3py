from .stochastic import EllipticalProcess, CopulaProcess
from .gaussian import GaussianProcess
from .studentT import StudentTProcess


class CopulaGaussianProcess(CopulaProcess):
    pass


class CopulaStudentTProcess(CopulaProcess):
    pass



class TransformedStudentTProcess(CopulaProcess):
    pass

from ..bayesian.models import TheanoBlackBox


class StochasticProcess(TheanoBlackBox):
    pass


class EllipticalProcess(StochasticProcess):
    pass


class CopulaProcess(StochasticProcess):
    pass

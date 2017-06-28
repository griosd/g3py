import datetime
import sys
from . import config
from .version import __version__
from .bayesian.models import *
from .libs import *


def version(file=None):
    if file is not None:
        file = open(file, 'w')
    import matplotlib
    import emcee
    import sklearn
    import numpy as np, scipy as sp, pandas as pd, seaborn as sb, theano as th, pymc3 as pm
    print('last execute', datetime.datetime.now(), file=file)
    print('python', sys.version.replace('\n', ' '), file=file)
    print('g3py', __version__, file=file)
    print(np.__name__, np.__version__, file=file)
    print(sp.__name__, sp.__version__, file=file)
    print(pd.__name__, pd.__version__, file=file)

    print(th.__name__, th.__version__, file=file)
    print(pm.__name__, pm.__version__, file=file)
    print(emcee.__name__, emcee.__version__, file=file)
    print(sklearn.__name__, sklearn.__version__, file=file)
    print(matplotlib.__name__, matplotlib.__version__, file=file)
    print(sb.__name__, sb.__version__, file=file)


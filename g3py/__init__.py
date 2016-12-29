from g3py.processes.stochastic import *
from g3py.processes.gaussian import *
from g3py.functions.hypers import *
from g3py.functions.kernels import *
from g3py.functions.mappings import *
from g3py.functions.means import *
from g3py.functions.metrics import *
from g3py.libs.data import *
from g3py.libs.plots import *
from g3py.libs.tensors import *
from g3py.libs.traces import *

from .models import *
from .tgp import *


th.config.floatX = 'float32'
th.config.on_unused_input = 'ignore'
th.config.mode = 'FAST_RUN'

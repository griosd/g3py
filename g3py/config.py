import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import theano as th
th.config.floatX = 'float32'
th.config.on_unused_input = 'ignore'
th.config.mode = 'FAST_RUN'
th.config.lib.amdlibm = False
th.config.warn_float64 = 'raise'
#th.config.cast_policy = 'numpy+floatX'
#th.config.int_division = 'raise'


plot_big = False
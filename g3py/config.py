import os
import theano as th

th.config.floatX = 'float32'
th.config.on_unused_input = 'ignore'
th.config.mode = 'FAST_RUN'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
#th.config.warn_float64 = 'warn'
#th.config.cast_policy = 'numpy+floatX'
#th.config.int_division = 'raise'

plot_big = False
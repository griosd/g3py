import theano as th

th.config.floatX = 'float32'
th.config.on_unused_input = 'ignore'
th.config.mode = 'FAST_RUN'

plot_big = False
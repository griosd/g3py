from g3py import StochasticProcess, Mapping, Identity, Kernel, Mean


class StudentTProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, noise=True, freedom=None,
                 name=None, inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=Identity(), noise=noise,
                         freedom=freedom, name=name, inputs=inputs, outputs=outputs, hidden=hidden)


class TransformedStudentTProcess(StochasticProcess):
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, mapping: Mapping=None, noise=True,
                 freedom=None, name=None, inputs=None, outputs=None, hidden=None):
        super().__init__(space=space, location=location, kernel=kernel, mapping=mapping, noise=noise,
                         freedom=freedom, name=name, inputs=inputs, outputs=outputs, hidden=hidden)


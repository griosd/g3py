import theano as th
import theano.tensor as tt
from g3py.functions.hypers import Hypers


class Mean(Hypers):
    def __call__(self, x):
        pass


class Zero(Mean):
    def __call__(self, x):
        return 0.0


class Bias(Mean):
    def __init__(self, x=None, name=None, constant=None):
        super().__init__(x, name)
        self.constant = constant

    def check_hypers(self, parent=''):
        if self.constant is None:
            self.constant = Hypers.Flat(parent+self.name+'_Constant')
        self.hypers += [self.constant]

    def default_hypers(self, x=None, y=None):
        return {self.constant: y.mean().astype(th.config.floatX)}

    def __call__(self, x):
        return self.constant


class Linear(Mean):
    def __init__(self, x=None, name=None, constant=None, coeff=None):
        super().__init__(x, name)
        self.constant = constant
        self.coeff = coeff

    def check_hypers(self, parent=''):
        if self.constant is None:
            self.constant = Hypers.Flat(parent+self.name+'_Constant')
        if self.coeff is None:
            self.coeff = Hypers.Flat(parent+self.name+'_Coeff', shape=self.shape)
        self.hypers += [self.constant, self.coeff]

    def default_hypers(self, x=None, y=None):
        return {self.constant: y.mean().astype(th.config.floatX),
                self.coeff: y.mean()/x.mean()}

    def __call__(self, x):
        return self.constant + tt.dot(x, self.coeff)

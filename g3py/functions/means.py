import theano as th
import theano.tensor as tt
from g3py.functions.hypers import Hypers


class Mean(Hypers):
    def __mul__(self, other):
        if issubclass(type(other), Mean):
            return MeanProd(self, other)
        else:
            return MeanScale(self, other)
    __imul__ = __mul__
    __rmul__ = __mul__

    def __add__(self, other):
        if issubclass(type(other), Mean):
            return MeanSum(self, other)
        else:
            return MeanShift(self, other)
    __iadd__ = __add__
    __radd__ = __add__

    def eval(self, x):
        pass

    def __call__(self, x):
        return self.eval(x[:, self.dims])


class BlackBox(Mean):
    def __init__(self, element, x=None, name=None):
        super().__init__(x, name)
        self.element = element

    def eval(self, x):
        return self.element

    def __call__(self, x):
        return self.element


class MeanOperation(Mean):
    def __init__(self, _m: Mean, _element):
        self.m = _m
        self.element = _element
        self.hypers = []

    def check_hypers(self, parent=''):
        self.m.check_hypers(parent=parent)
        self.hypers = self.m.hypers

    def check_dims(self, x=None):
        self.m.check_dims(x)

    def default_hypers_dims(self, x=None, y=None):
        return self.m.default_hypers_dims(x, y)

    def __str__(self):
        return str(self.element) + " op " + str(self.m)


class MeanComposition(Mean):
    def __init__(self, _m1: Mean, _m2: Mean):
        self.m1 = _m1
        self.m2 = _m2

    def check_hypers(self, parent=''):
        self.m1.check_hypers(parent=parent)
        self.m2.check_hypers(parent=parent)
        self.hypers = self.m1.hypers + self.m2.hypers

    def check_dims(self, x=None):
        self.m1.check_dims(x)
        self.m2.check_dims(x)

    def default_hypers_dims(self, x=None, y=None):
        return {**self.m1.default_hypers_dims(x, y), **self.m2.default_hypers_dims(x, y)}

    def __str__(self):
        return str(self.m1) + " op " + str(self.m2)


class MeanScale(MeanOperation):
    def __call__(self, x):
        return self.element * self.m(x)

    def __str__(self):
        return str(self.element) + " * " + str(self.m)


class MeanShift(MeanOperation):
    def __call__(self, x):
        return self.element + self.m(x)

    def __str__(self):
        return str(self.element) + " * " + str(self.m)


class MeanProd(MeanComposition):
    def __call__(self, x):
        return self.m1(x) * self.m2(x)

    def __str__(self):
        return str(self.m1) + " * " + str(self.m2)


class MeanSum(MeanComposition):
    def __call__(self, x):
        return self.m1(x) + self.m2(x)

    def __str__(self):
        return str(self.m1) + " * " + str(self.m2)


class Zero(Mean):
    def eval(self, x):
        return tt.zeros(shape=(x.shape[0],)) #TODO: check dims


class Bias(Mean):
    def __init__(self, x=None, name=None, bias=None):
        super().__init__(x, name)
        self.bias = bias

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.bias is None:
            self.bias = Hypers.Flat(parent + self.name + '_Bias')
        self.hypers += [self.bias]

    def default_hypers(self, x=None, y=None):
        return {self.bias: y.mean().astype(th.config.floatX)}

    def eval(self, x):
        return self.bias * tt.ones(shape=(x.shape[0],)) #TODO: check dims


class Linear(Mean):
    def __init__(self, x=None, name=None, constant=None, coeff=None):
        super().__init__(x, name)
        self.constant = constant
        self.coeff = coeff

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.constant is None:
            self.constant = Hypers.Flat(parent+self.name+'_Constant')
        if self.coeff is None:
            self.coeff = Hypers.Flat(parent+self.name+'_Coeff', shape=self.shape)
        self.hypers += [self.constant, self.coeff]

    def default_hypers(self, x=None, y=None):
        return {self.constant: y.mean().astype(th.config.floatX),
                self.coeff: y.mean()/x.mean(axis=0)}

    def eval(self, x):
        return self.constant + tt.dot(x, self.coeff) #TODO: check dims



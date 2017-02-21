from unit import unit
from math import exp

class sigmoidGate:
    def sig(self, x):
        return 1 / (1 + exp(-x))
    def forward(self, u0):
        self.u0 = u0;
        self.utop = unit(self.sig(self.u0.value), 0.0)
        return self.utop

    def backward(self):
        s = self.sig(self.u0.value)
        self.u0.grad += (s * (1-s)) * self.utop.grad;

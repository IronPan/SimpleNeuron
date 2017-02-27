from gate.unit import unit
from gate.addGate import addGate
from gate.multiplyGate import multiplyGate


class circuit:

    def __init__(self):
        self.mulg0 = multiplyGate()
        self.mulg1 = multiplyGate()
        self.addg0 = addGate()
        self.addg1 = addGate()

    def forward(self, x, y, a, b, c):
        self.ax = self.mulg0.forward(a, x)
        self.by = self.mulg1.forward(b, y)
        self.axpby = self.addg0.forward(self.ax, self.by)
        self.axpbypc = self.addg1.forward(self.axpby, c)
        return self.axpbypc

    def backward(self, gradient_top):
        self.axpbypc.grad = gradient_top
        self.addg1.backward()
        self.addg0.backward()
        self.mulg1.backward()
        self.mulg0.backward()


class svmGate:

    def __init__(self):
        self.a = unit(1, 0)
        self.b = unit(-2, 0)
        self.c = unit(-1, 0)
        self.circuit = circuit()

    def forward(self, x, y):
        self.unit_out = self.circuit.forward(x, y, self.a, self.b, self.c)
        return self.unit_out

    def backward(self, label):
        self.a.grad = 0
        self.b.grad = 0
        self.c.grad = 0
        pull = 0
        if label == 1 and self.unit_out.value < 0:
            pull = 1
        if label == -1 and self.unit_out.value > 0:
            pull = -1
        self.circuit.backward(pull)
        self.a.grad -= self.a.value
        self.b.grad -= self.b.value

    def parameterUpdate(self):
        step_size = 0.01
        self.a.value += step_size * self.a.grad
        self.b.value += step_size * self.b.grad
        self.c.value += step_size * self.c.grad

    def learnFrom(self, x, y, label):
        self.forward(x, y)
        self.backward(label)
        self.parameterUpdate()

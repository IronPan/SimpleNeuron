from gate.unit import unit

class addGate:
    def forward(self, u0, u1):
        self.u0 = u0;
        self.u1 = u1;
        self.utop = unit(u0.value + u1.value, 0.0)
        return self.utop

    def backward(self):
        self.u0.grad += 1 * self.utop.grad;
        self.u1.grad += 1 * self.utop.grad;


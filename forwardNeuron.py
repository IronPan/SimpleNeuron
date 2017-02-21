from unit import unit
from addGate import addGate
from multiplyGate import multiplyGate
from sigmoidGate import sigmoidGate

a = unit(1,0)
b = unit(2,0)
c = unit(-3,0)
x = unit(-1,0)
y = unit(3,0)

mulg0 = multiplyGate()
mulg1 = multiplyGate()
addg0 = addGate()
addg1 = addGate()
sg0 = sigmoidGate()

def forwardNeuron():
    ax = mulg0.forward(a,x)
    by = mulg1.forward(b,y)
    axpby = addg0.forward(ax, by)
    axpbypc = addg1.forward(axpby, c)
    s = sg0.forward(axpbypc)
    return s

s = forwardNeuron()
print(s.value)

s.grad = 1.0

sg0.backward()
addg1.backward()
addg0.backward()
mulg1.backward()
mulg0.backward()
print(str(a.grad) + ' ' + str(b.grad) + ' ' + str(c.grad) + ' ' + str(x.grad) + ' ' + str(y.grad))

step_size = 0.01
a.value += step_size * a.grad
b.value += step_size * b.grad 
c.value += step_size * c.grad 
x.value += step_size * x.grad
y.value += step_size * y.grad

s = forwardNeuron()
print(s.value)



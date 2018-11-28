
import random
from gate.unit import unit
from gate.svmGate import svmGate

data = [[1.2, 0.7], [-0.3, -0.5], [3.0, 0.1],
        [-0.1, -1.0], [-1.0, 1.1], [2.1, -3]]
labels = [1, -1, 1, -1, -1, 1]

svmGate = svmGate()


def evalTrainingAccuracy():
    num_correct = 0
    for i in range(0, len(data)):
        x = unit(data[i][0], 0)
        y = unit(data[i][1], 0)
        true_label = labels[i]
        predicted_label = 1 if svmGate.forward(x, y).value > 0 else -1
        if true_label == predicted_label:
            num_correct += 1
    return num_correct / float(len(data))


for r in range(0, 2000):
    i = int(random.random() * len(data))
    x = unit(data[i][0], 0)
    y = unit(data[i][1], 0)
    label = labels[i]
    if r % 25 == 0:
        print('training accuracy at iter ' + str(r) +
              ': ' + str(evalTrainingAccuracy()))
    svmGate.learnFrom(x, y, label)

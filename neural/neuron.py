import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron(object):
    def __init__(self, b, w):
        self.b = b
        self.w = w

    def forward(self, x):
        y = np.dot(self.w, x) + b
        return sigmoid(y)

Xi = np.array([2, 3])
Wi = np.array([0, 1])
b = 4
n = Neuron(b, Wi)
print('Y=', n.forward(Xi))
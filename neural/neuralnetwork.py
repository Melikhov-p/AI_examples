import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron(object):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, x):
        y = np.dot(self.weights, x) + self.bias
        return sigmoid(y)


class NeuralNetwork(object):
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # Формируем сеть
        # Скрытый слой
        self.h1 = Neuron(weights, bias)  # Скрытый нейрон 1
        self.h2 = Neuron(weights, bias)  # Скрытый нейрон 2
        # Выходной слой
        self.o1 = Neuron(weights, bias)  # Выходной нейрон

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)  # Выход нейрона h1
        out_h2 = self.h2.feedforward(x)  # Выход нейрона h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


network = NeuralNetwork()
# Входные данные | Входной слой
x = np.array([2, 3])

print('Y=', network.feedforward(x))
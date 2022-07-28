import numpy as np
import scipy
import pybrain3
from pybrain3.supervised import BackpropTrainer
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
import matplotlib.pyplot as plt
import pickle

net = buildNetwork(2, 3, 1)
ds = SupervisedDataSet(2, 1)

xorModel = [
    [(0,0), (0,)],
    [(0,1), (1,)],
    [(1,0), (1,)],
    [(1,1), (0,)],
]

for input, target in xorModel:
    ds.addSample(input, target)

trainer = BackpropTrainer(net)
trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)
with open('model.txt', 'wb') as file:
    pickle.dump(net, file)
    file.close()

file = open('model.txt', 'rb')
net2 = pickle.load(file)

print(net2.activate([1, 0]))
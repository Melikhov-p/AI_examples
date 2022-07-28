# Нейросеть рекомендации похода на рыбалку

import pybrain3
import pickle
import matplotlib.pyplot as plt
from numpy import ravel
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(4, 1)
data = [
    [(2, 3, 80, 1), (5,)],
    [(3, 4, 70, 1), (5,)],
    [(4, 4, 60, 1), (5,)],
    [(5, 5, 50, 2), (4,)],
    [(7, 6, 45, 2), (4,)],
    [(8, 6, 43, 2), (4,)],
    [(10, 7, 40, 3), (3,)],
    [(12, 8, 30, 3), (3,)],
    [(14, 8, 25, 3), (3,)],
    [(15, 9, 20, 4), (2,)],
    [(17, 9, 15, 4), (2,)],
    [(18, 10, 12, 4), (2,)],
    [(20, 11, 10, 5), (1,)],
    [(25, 15, 5, 6), (1,)],
    [(30, 20, 3, 7), (1,)],
]

for input, target in data:
    ds.addSample(input, target)

net = buildNetwork(4, 8, 1, bias=True)

trainer = BackpropTrainer(net, dataset=ds, verbose=True, momentum=0.0, learningrate=0.01, weightdecay=0.01)
trnerr, valerr = trainer.trainUntilConvergence(maxEpochs=2500)
plt.plot(trnerr, 'b', valerr, 'r')
plt.show()
print('Y1 =', net.activate([2, 3, 80, 1]))

# with open('NetFishing.txt', 'wb') as file:
#     pickle.dump(net, file)
#
# print('Y1 = ', net.activate([2, 3, 80, 1]))
# with open('NetFishing.txt', 'rb') as file:
#     net2 = pickle.load(file)
#     print('Y2 = ', net2.activate([20, 11, 80, 1]))

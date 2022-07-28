import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AdaptiveLinearNeuron(object):
    def __init__(self, rate, n_iter):
        self.rate = rate # rate - темп обучения
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(1+X.shape[1])
        self.cost = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.weights[1:] += self.rate * X.T.dot(errors)
            self.weights[0] += self.rate * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost.append(cost)
        return self

    def net_input(self, X): # чистый вход
        return np.dot(X, self.weights[1:] + self.weights[0])

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

# загрузка данных для обучения
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv('Iris.csv', header=None)
# print('Данные об ирисах: ')
# print(df.to_string())
# df.to_csv('Iris.csv')

X = df.iloc[0:100, [0, 2]] # Выборка первых 100 строк массива (столбец 0 и 2)
# print('Значение X - 100')
# print(X)

y = df.iloc[0:100, 4] # Выборка первых 100 строк массива (столбец 4 - название цветка)
y = np.where(y == 'Iris-setosa', -1, 1)
# print('Значение названий цветков в виде -1 и 1')
# print(y)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
# rate = 0.0001
a1n1 = AdaptiveLinearNeuron(0.0001, 10).fit(X,y)
ax.plot(range(1, len(a1n1.cost)+1), np.log10((a1n1.cost)), marker='o')
ax.set_xlabel('Эпохи')
ax.set_ylabel('log(Сумма квадратичных ошибок)')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class perceptron(object):
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y): # Функция тренировки перцептрона
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X.values, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X): # чистый вход
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X): # вернуть предсказанную метку класса
        return np.where(self.net_input(X) >= 0.0, 1, -1)


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


# Первые 50 элементов обучающей выборки (строки 0-50, столбцы О, 1)
plt.scatter(X[0][0:50], X[2][0:50], color='red', marker='o', label='щетинистый')
# Следующие 50 элементов обучающей выборки (строки 50-100, столбцы·о, 1)
plt.scatter(X[0][50:100], X[2][50:100], color='blue', marker='x', label='разноцветный')
# Формирование названий осей и вывод графика на экран
plt.xlabel('Длина чашелистика')
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()

# Тренировка
ppn = perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Ошибки')
plt.show()

i1=[5.5, 1.6]
i2=[6.4, 4.5]
R1 = ppn.predict(i1)
R2 = ppn.predict(i2)
print('R1=', R1,' R2=', R2)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# df = sb.load_dataset('iris')
# sb.set_style('ticks')
# sb.pairplot(df, hue='species', diag_kind='kde', kind='scatter', palette='husl')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
z = knn.fit(X_train, y_train)

print(knn.predict(np.array([[6.1, 2.9, 4.7, 1.4]])))
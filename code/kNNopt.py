from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

# zur Veranschaulichung wieder nur 2 Features (sepal_length und sepal_width)
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=142)

# Das optimale k ermitteln
error_rate = []
kMin, kMax = 1, 15

for i in range(kMin, kMax+1):
    kNN = KNeighborsClassifier(n_neighbors=i)
    kNN.fit(X_train, y_train)
    y_pred = kNN.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))

# Die Ergebnisse ausgeben
plt.figure(figsize=(12, 6))
plt.plot(range(kMin, kMax+1), error_rate, marker="o", markerfacecolor="green",
         linestyle="dashed", color="red", markersize=15)
plt.title("Error rate vs k value", fontsize=20)
plt.xlabel("k- values", fontsize=20)
plt.ylabel("error rate", fontsize=20)
plt.xticks(range(kMin, kMax+1))
plt.show()

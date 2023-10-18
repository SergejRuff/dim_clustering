from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import kNNplot

# Datensatz laden
iris = load_iris()

# Schlüssel anzeigen (Datenstruktur ist Dictionary)
print(iris.keys())

# Auswählen und Anzeigen der Features
X = iris.data
print(iris.feature_names)
print(iris.data.tolist())

# Auswählen und Anzeigen der Zielvariablen
y = iris.target
print(iris.target_names)
print(iris.target)

# Datensatz aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=142)
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# Den kNN-Klassifikator aussuchen und anwenden
kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(X_train, y_train)

# Ausgabe der zu den einzelnen Datenpunkten zugeordneten Klassen
y_pred = kNN.predict(X_test)
print(y_pred)
# direkter Vergleich mit den tatsächlichen Klassen
print(y_test)

# Beispiel: Ausgabe der Klassenzugehörigkeit eines Datenpunktes
print(iris.target_names[y_pred[14]])
# direkter Vergleich mit der tatsächlichen Klasse
print(iris.target_names[y_test[14]])

# Beispiel: Ausgabe der Wahrscheinlichkeiten der Klassenzugehörigkeit
# hier: 60% virginica, 40% versicolor
print(kNN.predict_proba(X_test)[14])

# Erzeugen und Anzeigen der Konfusionsmatrix
print(confusion_matrix(y_test, y_pred))

# Die Genauigkeit ermitteln und ausgeben
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

# Die Klassifikation über 2 Parameter (sepal_length und sepal_width)
# jeweils mit k = 1, 5, 10, 15
X = iris.data[:, :2]

kNN = KNeighborsClassifier(n_neighbors=1)
kNN.fit(X, y)
kNNplot.plot_iris_knn(kNN, X, y)

kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(X, y)
kNNplot.plot_iris_knn(kNN, X, y)

kNN = KNeighborsClassifier(n_neighbors=10)
kNN.fit(X, y)
kNNplot.plot_iris_knn(kNN, X, y)

kNN = KNeighborsClassifier(n_neighbors=15)
kNN.fit(X, y)
kNNplot.plot_iris_knn(kNN, X, y)

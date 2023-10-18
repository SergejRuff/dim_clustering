from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import kMplot

# Datensatz laden
iris = load_iris()

# Auswählen und Anzeigen der Features
X = iris.data
print(iris.feature_names)
print(iris.data.tolist())

# Den kNN-Klassifikator aussuchen und anwenden
kM = KMeans(n_clusters=3)
kM.fit(X)

# Die Cluster ausgeben
print(kM.labels_) # kM.predict(X)

# die Clusterzentren ausgeben
print(kM.cluster_centers_)

# Das Clustering über 2 Parameter (sepal_length und sepal_width)
# jeweils mit k = 1, 5, 10, 15
X = iris.data[:, :2]

kM = KMeans(n_clusters=3)
kM.fit(X)

print(kM.labels_)
print(kM.cluster_centers_)

kMplot.plot_iris_kM(kM, X)


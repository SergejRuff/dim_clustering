from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

# Das optimale k ermitteln
elbow, ss = [], []
kMin, kMax = 2, 10

for i in range(kMin, kMax+1):
   kM = KMeans(n_clusters=i)
   y_pred = kM.fit_predict(X)
   silhouette_avg = silhouette_score(X, y_pred)
   ss.append(silhouette_avg)
   elbow.append(kM.inertia_)

# Die Ergebnisse ausgeben
fig = plt.figure(figsize=(14, 7))
fig.add_subplot(121)
plt.plot(range(kMin, kMax+1), elbow, 'b-', label='Sum of squared distances')
plt.xlabel("Number of cluster")
plt.ylabel("SSD")
plt.legend()
fig.add_subplot(122)
plt.plot(range(kMin, kMax+1), ss, 'b-', label='Silhouette Score')
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette Score")
plt.legend()
plt.show()


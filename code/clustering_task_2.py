"""
exercise demonstating clustering methods k means clustering and BDSCAN
and dimensional reduction (PCA)
"""

# Import modules
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from perform_pca import perform_pca
import numpy as np


titanic_train = pd.read_csv("data/train.csv")
titanic_test = pd.read_csv("data/test.csv")
y_test = pd.read_csv("data/gender_submission.csv")
y_test = y_test.drop("PassengerId", axis=1)
# filtering train

# replace male and female with integer values
titanic_train = titanic_train.replace(["male", "female"], [0, 1])

# drop unwanted coloumns

titanic_train = titanic_train.drop(
    ["Cabin", "Ticket", "Name", "Embarked", "PassengerId"], axis=1)

titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].mean())

train_x = titanic_train.drop("Survived", axis=1)
train_y = titanic_train["Survived"]

# filtering test


# replace male and female with integer values
titanic_test = titanic_test.replace(["male", "female"], [0, 1])

# drop unwanted coloumns

titanic_test = titanic_test.drop(
    ["Cabin", "Ticket", "Name", "Embarked", "PassengerId"], axis=1)

titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].mean())

# has one NA that needs to be filled.
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].mean())


silhouette_scores_kmeans = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(train_x)
    silhouette_avg = silhouette_score(train_x, cluster_labels)
    silhouette_scores_kmeans.append(silhouette_avg)

silhouette_scores_dbscan = []
eps_values = [0.5, 1.0, 1.5]
min_samples_values = [5, 10, 15]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(train_x)
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(train_x, cluster_labels)
            silhouette_scores_dbscan.append((eps, min_samples, silhouette_avg))

# Plot K-Means Silhouette Scores
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_range, silhouette_scores_kmeans, marker='o')
plt.title('K-Means Silhouette Scores')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

# Plot DBSCAN Silhouette Scores
plt.subplot(1, 2, 2)
for eps, min_samples, silhouette_avg in silhouette_scores_dbscan:
    label = f'eps={eps}, min_samples={min_samples}'
    plt.plot(label, silhouette_avg, marker='o')
plt.title('DBSCAN Silhouette Scores')
plt.xlabel('Parameters (eps, min_samples)')
plt.ylabel('Silhouette Score')
plt.xticks(rotation=90)  # Rotate x-labels for readability

plt.tight_layout()
plt.savefig("output/silscore.png")
plt.show()

# task 2 b

# Apply PCA to reduce the data to two principal components (PC1 and PC2)
train_x_pca = perform_pca(train_x, 2)

# Apply kMeans clustering with k=2
kmeans = KMeans(n_clusters=2, n_init=10, max_iter=300)
kmeans_labels = kmeans.fit_predict(train_x_pca)

# Apply DBScan clustering
eps = 0.5
min_samples = 10
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(train_x_pca)


# Plot the results
plt.figure(figsize=(12, 5))

# Plot kMeans clustering results
plt.subplot(1, 2, 1)
unique_kmeans_labels = np.unique(kmeans_labels)
colors = ['r', 'c', 'm']  # You can add more colors if needed
for i, label in enumerate(unique_kmeans_labels):
    plt.scatter(train_x_pca[kmeans_labels == label, 0],
                train_x_pca[kmeans_labels == label, 1], c=colors[i], label=f'Cluster {label} (kMeans)')
plt.title('kMeans Clustering (k=2)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# Plot DBScan clustering results
plt.subplot(1, 2, 2)
unique_dbscan_labels = np.unique(dbscan_labels)
for i, label in enumerate(unique_dbscan_labels):
    plt.scatter(train_x_pca[dbscan_labels == label, 0], train_x_pca[dbscan_labels ==
                label, 1], c=colors[i], label=f'Cluster {label} (DBScan)')
plt.title('DBScan Clustering (eps={}, min_samples={})'.format(eps, min_samples))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.tight_layout()

plt.savefig("output/kclustervsdbscan.png")

plt.show()

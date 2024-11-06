from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
X = data.data

# Normalizacija, standartizacija ....

wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel("klasteriu sk.")
plt.ylabel("nuostolis")
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k)
y_kmeans = kmeans.fit_predict(X)

# Braizome
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
plt.xlabel('Ilgis')
plt.ylabel('Plotis')
plt.show()
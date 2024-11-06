#Tikslas: Klasterizuoti Breast Cancer duomenų rinkinį.
#Instrukcijos:
#1. Įkelkite Breast Cancer duomenų rinkinį naudodami sklearn.datasets.load_breast_cancer.
#2. Standartizuokite duomenų rinkinį.
#3. Nubraižykite dendrogramą ir nustatykite tinkamą aukštį, kuriame būtų galima atskirti klasterius.
#4. Apskaičiuokite klasterių sumą kvadratu (WCSS) kiekvienam klasterių skaičiui ir nubraižykite Alkūnės metodą.
#5. Pritaikykite agglomerative klasterizavimą su optimaliausiu k ir skirtingais sujungimo metodais.
#6. Palyginkite klasterių etiketes su tikromis etiketėmis (gerybinis ir piktybinis).

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Įkeliame Breast Cancer duomenų rinkinį
data = load_breast_cancer()
X = data.data
y_true = data.target

# 2. Standartizuojame duomenų rinkinį
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Nubraižome dendrogramą ir nustatome tinkamą aukštį klasteriams
plt.figure(figsize=(10, 7))
linked = linkage(X_scaled, method='ward')
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering on Breast Cancer Dataset")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# 4. Apskaičiuojame klasterių sumą kvadratu (WCSS) ir naudojame Alkūnės metodą
wcss = []
max_clusters = 10
for k in range(1, max_clusters + 1):
    agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agglomerative.fit_predict(X_scaled)

    centroids = []
    for cluster_id in np.unique(labels):
        cluster_points = X_scaled[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    _, dists = pairwise_distances_argmin_min(X_scaled, centroids)
    wcss.append(np.sum(dists ** 2))

# Nubraižome Alkūnės metodą
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal K in Agglomerative Clustering on Breast Cancer Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 5. Pritaikome agglomerative klasterizavimą su optimaliausiu k ir skirtingais sujungimo metodais
optimal_clusters = 2  # Remiantis Alkūnės metodu
linkages = ['ward', 'complete', 'average', 'single']
for linkage_method in linkages:
    clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage=linkage_method)
    labels = clustering.fit_predict(X_scaled)
    print(f"Linkage: {linkage_method}, Labels: {labels}")

    # 6. Palyginam klasterių etiketes su tikromis etiketėmis
    accuracy_original = accuracy_score(y_true, labels)

    # Tikriname tikslumą su apverstomis etiketėmis
    labels_inverted = np.where(labels == 0, 1, 0)
    accuracy_inverted = accuracy_score(y_true, labels_inverted)

    print(f"Accuracy with original labels: {accuracy_original:.2f}")
    print(f"Accuracy with inverted labels: {accuracy_inverted:.2f}")

    # Pasirenkame geresnį tikslumą
    if accuracy_inverted > accuracy_original:
        print("Inverted labels give a better match to true labels.")
    else:
        print("Original labels give a better match to true labels.")

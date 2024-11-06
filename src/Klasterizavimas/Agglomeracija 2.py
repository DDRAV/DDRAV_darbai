#Tikslas: Klasterizuoti Vyno duomenų rinkinį.
#Instrukcijos:
#1. Įkelkite Vyno duomenų rinkinį naudodami sklearn.datasets.load_wine.
#2. Standartizuokite duomenų rinkinį.
#3. Nubraižykite dendrogramą ir nustatykite tinkamą aukštį, kuriame būtų galima atskirti klasterius.
#4. Apskaičiuokite klasterių sumą kvadratu (WCSS) kiekvienam klasterių skaičiui ir nubraižykite Alkūnės metodą.
#5. Pritaikykite agglomerative klasterizavimą su optimaliausiu k ir skirtingais sujungimo metodais.
#6. Palyginkite klasterių etiketes su tikromis etiketėmis

from itertools import permutations
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Įkeliame Vyno duomenų rinkinį
data = load_wine()
X = data.data
y_true = data.target

# 2. Standartizuojame duomenų rinkinį
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Nubraižome dendrogramą ir nustatome tinkamą aukštį klasteriams
plt.figure(figsize=(10, 7))
linked = linkage(X_scaled, method='ward')
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering on Wine Dataset")
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

# Nubraizom Alkūnės metodą
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal K in Agglomerative Clustering on Wine Dataset')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 5. Pritaikom agglomerative klasterizavimą su optimaliausiu k ir skirtingais sujungimo metodais
optimal_clusters = 3  # Nustatykite optimalų klasterių skaičių pagal Alkūnės metodą arba dendrogramą
linkages = ['ward', 'complete', 'average', 'single']
for linkage_method in linkages:
    clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage=linkage_method)
    labels = clustering.fit_predict(X_scaled)
    print(f"Linkage: {linkage_method}, Labels: {labels}")

    # 6. Palyginame klasterių etiketes su tikromis etiketėmis
    # Tikriname tikslumą su pradinėmis ir apverstomis etiketėmis

    accuracy = accuracy_score(y_true, labels)
    print(f"Accuracy with {linkage_method} linkage: {accuracy:.2f}")

    label_permutations = list(permutations([0, 1, 2]))
    best_accuracy = 0
    best_permutation = None

    for perm in label_permutations:
        permuted_labels = np.array([perm[label] for label in labels])
        accuracy = accuracy_score(y_true, permuted_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_permutation = perm

    print(f"Best accuracy with {linkage_method} linkage: {best_accuracy:.2f} with permutation {best_permutation}")
    # accuracy_original = accuracy_score(y_true, labels)
    # labels_inverted = np.where(labels == 0, 1, np.where(labels == 1, 0, labels))
    # accuracy_inverted = accuracy_score(y_true, labels_inverted)

    # print(f"Accuracy with {linkage_method} linkage - Original Labels: {accuracy_original:.2f}")
    # print(f"Accuracy with {linkage_method} linkage - Inverted Labels: {accuracy_inverted:.2f}")

    # Pasirinkite didesnį tikslumą
    # if accuracy_inverted > accuracy_original:
    # print(f"Inverted labels give a better match for {linkage_method} linkage.")
    # else:
    # print(f"Original labels give a better match for {linkage_method} linkage.")
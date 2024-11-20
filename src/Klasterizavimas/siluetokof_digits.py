from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Įkeliame 'load_digits' duomenų rinkinį
data = load_digits()
X = data.data
y = data.target

# 2. Pervadiname į DataFrame formatą ir peržiūrime duomenų struktūrą
digits_df = pd.DataFrame(X)
digits_df['Target'] = y

# Perviūrėkime pirmas 10 eilutes
print(digits_df.columns)
print(digits_df.head(10))
print(digits_df['Target'])
print(digits_df['Target'].unique())
print(digits_df['Target'].value_counts())

# 3. Standartizuojame duomenis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Funkcija klasterizavimo rezultatų atvaizdavimui
def plot_silhouette(method_name, labels, silhouette_vals, overall_silhouette_score, optimal_clusters):
    plt.figure(figsize=(10, 6))

    # Atitinkami intervalai kiekvienam klasteriui
    y_lower = 10
    for i in range(optimal_clusters):
        cluster_vals = silhouette_vals[labels == i]
        cluster_vals.sort()  # Rūšiuojame klasterių reikšmes
        y_upper = y_lower + len(cluster_vals)

        # Pildome grafiko plotą
        plt.fill_betweenx(range(y_lower, y_upper), 0, cluster_vals, alpha=0.7, label=f'Cluster {i}')

        # Nustatome žemesnę reikšmę kito klasterio atveju
        y_lower = y_upper

    # Piešiamas bendras silueto įverčio horizontalus linija
    plt.axvline(overall_silhouette_score, color="red", linestyle="--", label="Overall Silhouette Score")

    # Gražus pavadinimas ir etiketės
    plt.title(f"Silhouette Score per Cluster for {method_name}")
    plt.xlabel("Silhouette Score")
    plt.ylabel("Data Points")
    plt.legend()
    plt.show()

# Alkūnės metodas optimaliam klasterių skaičiui nustatyti
wcss = []
max_clusters = 20
for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# # Nubraižome Alkūnės metodą
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, max_clusters + 1), wcss, marker='o')
# plt.title('Elbow Method for Optimal K in KMeans Clustering on Digits Dataset')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.show()

# # Sukuriame ir nubraižome dendrogramą
# plt.figure(figsize=(10, 6))
# Z = linkage(X_scaled, method='ward')
# dendrogram(Z)
# plt.title("Dendrogram for Agglomerative Clustering on Digits Dataset")
# plt.xlabel("Data points")
# plt.ylabel("Distance")
# plt.show()

# Pagal grafiką nustatykite optimalų klasterių skaičių
optimal_clusters = 10# Nustatyta pagal Alkūnės metodą

# 4. Naudojame KMeans klasterizavimą
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
overall_silhouette_score_kmeans = silhouette_score(X_scaled, labels_kmeans)
silhouette_vals_kmeans = silhouette_samples(X_scaled, labels_kmeans)

print(f"KMeans Overall Silhouette Score: {overall_silhouette_score_kmeans}")
kmeans_silhouette_df = pd.DataFrame({'Cluster': labels_kmeans, 'Silhouette Score': silhouette_vals_kmeans})
kmeans_cluster_silhouette_scores = kmeans_silhouette_df.groupby('Cluster')['Silhouette Score'].mean()
print("KMeans Silhouette Scores for Each Cluster:")
print(kmeans_cluster_silhouette_scores)

plot_silhouette("KMeans", labels_kmeans, silhouette_vals_kmeans, overall_silhouette_score_kmeans, optimal_clusters)

# 5. Naudojame Agglomerative Clustering su tuo pačiu klasterių skaičiumi
clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
labels_agg = clustering.fit_predict(X_scaled)
overall_silhouette_score_agg = silhouette_score(X_scaled, labels_agg)
silhouette_vals_agg = silhouette_samples(X_scaled, labels_agg)

print(f"Agglomerative Clustering Overall Silhouette Score: {overall_silhouette_score_agg}")
agg_silhouette_df = pd.DataFrame({'Cluster': labels_agg, 'Silhouette Score': silhouette_vals_agg})
agg_cluster_silhouette_scores = agg_silhouette_df.groupby('Cluster')['Silhouette Score'].mean()
print("Agglomerative Clustering Silhouette Scores for Each Cluster:")
print(agg_cluster_silhouette_scores)

plot_silhouette("Agglomerative Clustering", labels_agg, silhouette_vals_agg, overall_silhouette_score_agg, optimal_clusters)



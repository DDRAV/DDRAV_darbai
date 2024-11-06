from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# 1. Užkraunam Wine duomenų rinkinį
data = load_wine()
X = data.data
feature_names = data.feature_names

# Konvertuojame naudojant pandas paziurime duomenys
wine_df = pd.DataFrame(X, columns=feature_names)
pd.set_option('display.max_columns', None)
print("Wine duomenų rinkinys (pirmos 5 eilutės):")
print(wine_df.head())

# 2. Standartizuojame duomenis
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 3. Apskaičiuojame WCSS pagal klasterių skaičių ir braižome "alkūnės" grafiką
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_std)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Klasterių skaičius (k)")
plt.ylabel("WCSS")
plt.title("WCSS pagal klasterių skaičių")
plt.show()

# 4. Pasirenkam optimaliausia klasteriu kieki ir nubraizom grafika
optimal_k = 3  # Pasirinkite optimalų k pagal grafika
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans.fit_predict(X_std)

# Braižome klasterizavimo rezultatus
plt.scatter(X_std[:, 0], X_std[:, 1], c=y_kmeans, cmap='viridis', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.xlabel('Pirma Standartizuota Ašis')
plt.ylabel('Antra Standartizuota Ašis')
plt.title(f"KMeans Klasterizacija su {optimal_k} Klasteriais")
plt.show()

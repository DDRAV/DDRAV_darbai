# Silueto koeficientas yra matas, rodantis, kaip gerai atskirti klasteriai sprendime.
# Jis padeda nustatyti, ar klasteriai yra aiškiai apibrėžti.

# Formule:
# S = (b_i-a_i)/(max(b_i, a_i))
# i - duomenu taskas
# a_i - vidutinis atstumas taško į iki kitų kaimynų klasteryje
# b_i - vidutinis taško atstumas nuo taškų kitame, artimiausiaume klasteryje

#  1: Taškas gerai atitinka savo grupę ir blogai atitinka artimiausią kaimyninę grupę.
#  0: Taškas yra ties dviejų grupių riba.
# −1: Tikėtina, kad taškas priskirtas netinkamai.

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
X = data.data

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

overall_silhouette_score = silhouette_score(X, labels)
print(f"Overall Silhouette Score: {overall_silhouette_score}")

silhouette_vals = silhouette_samples(X, labels)
silhouette_df = pd.DataFrame({'Cluster': labels, 'Silhouette Score': silhouette_vals})
cluster_silhouette_scores = silhouette_df.groupby('Cluster')['Silhouette Score'].mean()

print("Silhouette Scores for Each Cluster:")
print(cluster_silhouette_scores)

# from sklearn.datasets import load_digits
# from sklearn.datasets import load_diabetes
# from sklearn.datasets import load_breast_cancer
# from sklearn.datasets import fetch_20newsgroups
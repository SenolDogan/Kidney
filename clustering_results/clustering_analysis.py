import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# Veri yükle
file_path = 'kidney.xlsx'
df = pd.read_excel(file_path)

# Temel sayısal ve kategorik değişkenleri seç (örnek: yaş, cinsiyet, eGFR, BMI, diyabet, vb.)
features = [
    'AGE_Baseline',
    'Sex_1male_2female',
    'Diabetes_status_1yes_0no',
    'eGFR_CKD_EPI_Creatinine_at_Baseline',
    'BMI'
]

# Eksik değerleri median ile doldur
for col in features:
    df[col] = df[col].fillna(df[col].median())

# Kategorik değişkenleri one-hot encode et
cat_cols = ['Sex_1male_2female', 'Diabetes_status_1yes_0no']
df[cat_cols] = df[cat_cols].astype('category')
df_encoded = pd.get_dummies(df[features], columns=cat_cols, drop_first=True)

# Standartlaştır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# PCA ile 2 boyuta indir
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans ile kümeleme (k=3 örnek)
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(7,5))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_kmeans, cmap='viridis', alpha=0.7)
plt.title('KMeans Clustering (PCA 2D)', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
cbar = plt.colorbar(scatter, label='Cluster')
plt.figtext(0.5, -0.08, 'Colors indicate cluster membership assigned by KMeans algorithm.', wrap=True, ha='center', fontsize=10)
plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig('clustering_results/kmeans_pca.png', bbox_inches='tight')
plt.close()

# Hiyerarşik kümeleme (dendrogram)
linked = linkage(X_scaled, 'ward')
plt.figure(figsize=(12, 6))
dend = dendrogram(
    linked,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=False,
    color_threshold=None,
    no_labels=True
)
plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.savefig('clustering_results/hierarchical_dendrogram.png', bbox_inches='tight')
plt.close()

# Hiyerarşik kümeleme ile etiketler (k=3 örnek)
hier_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X_scaled)
plt.figure(figsize=(7,5))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=hier_labels, cmap='plasma', alpha=0.7)

# Her küme için centroid hesapla ve kırmızı nokta ile göster
import numpy as np
for i in np.unique(hier_labels):
    centroid = X_pca[hier_labels == i].mean(axis=0)
    plt.scatter(centroid[0], centroid[1], c='red', s=200, marker='o', edgecolor='black', label=f'Cluster {i} centroid')

plt.title('Hierarchical Clustering (PCA 2D) + Centroids', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
cbar = plt.colorbar(scatter, label='Cluster')
plt.figtext(0.5, -0.08, 'Colors indicate cluster membership assigned by hierarchical clustering. Red dots: cluster centroids.', wrap=True, ha='center', fontsize=10)
plt.legend()
plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig('clustering_results/hierarchical_pca_with_centroids.png', bbox_inches='tight')
plt.close()

# KMeans PCA plotunda da kümelerin merkezlerini kırmızı nokta ile gösteren kodu ekleyeceğim ve yeni bir dosya olarak kaydedeceğim.
plt.figure(figsize=(7,5))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_kmeans, cmap='viridis', alpha=0.7)

# Her KMeans kümesi için centroid hesapla ve kırmızı nokta ile göster
for i in np.unique(labels_kmeans):
    centroid = X_pca[labels_kmeans == i].mean(axis=0)
    plt.scatter(centroid[0], centroid[1], c='red', s=200, marker='o', edgecolor='black', label=f'Cluster {i} centroid')

plt.title('KMeans Clustering (PCA 2D) + Centroids', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
cbar = plt.colorbar(scatter, label='Cluster')
plt.figtext(0.5, -0.08, 'Colors indicate cluster membership assigned by KMeans algorithm. Red dots: cluster centroids.', wrap=True, ha='center', fontsize=10)
plt.legend()
plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig('clustering_results/kmeans_pca_with_centroids.png', bbox_inches='tight')
plt.close()

# Küme özetleri
cluster_summary = pd.DataFrame(X_scaled, columns=df_encoded.columns)
cluster_summary['KMeans_Cluster'] = labels_kmeans
cluster_summary['Hierarchical_Cluster'] = hier_labels
summary = cluster_summary.groupby('KMeans_Cluster').mean()
summary.to_excel('clustering_results/kmeans_cluster_summary.xlsx')

print('Kümeleme analizi tamamlandı. Sonuçlar ve grafikler clustering_results klasörüne kaydedildi.') 
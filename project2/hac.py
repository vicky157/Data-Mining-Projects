import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_with_tsne(X_scaled, labels, title='t-SNE visualization'):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f'Cluster {label}', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.show()

def preprocess_and_cluster_hac(filename, linkage_method):
    df = pd.read_csv(filename, sep="\t", header=None)
    
    X = df.drop([0, 1], axis=1).values  
    ground_truth = df[1].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    n_clusters_options = range(2, 11)  
    best_silhouette_score = -1
    
    for n_clusters in n_clusters_options:
        hac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method).fit(X_scaled)
        labels = hac.labels_
        
        silhouette_avg = silhouette_score(X_scaled, labels)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters = n_clusters
            best_labels = labels
    
    print(f"Linkage: {linkage_method}, Best N Clusters: {best_n_clusters}, Silhouette Score: {best_silhouette_score}")
    
    rand_index = adjusted_rand_score(ground_truth, best_labels)
    print(f"Adjusted Rand Index: {rand_index}")
    
    plot_with_tsne(X_scaled, best_labels, title=f'HAC ({linkage_method}) Clustering with t-SNE ({filename})')

    return best_n_clusters, best_silhouette_score, rand_index

linkage_methods = ['ward', 'complete']

for linkage_method in linkage_methods:
    print(f"\nProcessing 'cho.txt' Dataset with {linkage_method} linkage")
    cho_n_clusters, cho_silhouette, cho_rand = preprocess_and_cluster_hac('cho.txt', linkage_method)

    print(f"\nProcessing 'iyer.txt' Dataset with {linkage_method} linkage")
    iyer_n_clusters, iyer_silhouette, iyer_rand = preprocess_and_cluster_hac('iyer.txt', linkage_method)

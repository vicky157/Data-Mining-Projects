import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
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

def preprocess_and_cluster_gmm(filename):
    df = pd.read_csv(filename, sep="\t", header=None)
    
    X = df.drop([0, 1], axis=1).values  
    ground_truth = df[1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components_options = range(2, 11)  
    best_silhouette_score = -1
    
    for n_components in n_components_options:
        gmm = GaussianMixture(n_components=n_components, random_state=42).fit(X_scaled)
        labels = gmm.predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, labels)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_components = n_components
            best_labels = labels
    
    print(f"Best N Components: {best_n_components}, Silhouette Score: {best_silhouette_score}")

    rand_index = adjusted_rand_score(ground_truth, best_labels)
    print(f"Adjusted Rand Index: {rand_index}")
    plot_with_tsne(X_scaled, best_labels, title=f'GMM Clustering with t-SNE ({filename})')

    return best_n_components, best_silhouette_score, rand_index
print("Processing 'cho.txt' Dataset")
cho_n_components, cho_silhouette, cho_rand = preprocess_and_cluster_gmm('cho.txt')

print("\nProcessing 'iyer.txt' Dataset")
iyer_n_components, iyer_silhouette, iyer_rand = preprocess_and_cluster_gmm('iyer.txt')

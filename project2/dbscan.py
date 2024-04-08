import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.impute import SimpleImputer
def plot_with_tsne(X_scaled, labels, title='t-SNE visualization'):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f'Cluster {label}' if label != -1 else 'Outliers', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.show()

def preprocess_and_cluster_dbscan(filename):
    df = pd.read_csv(filename, sep="\t", header=None)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    values = imputer.fit_transform(df.iloc[:, 2:])
    X = df.drop([0, 1], axis=1).values  
    ground_truth = df[1].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {'eps': [0.5, 1, 1.5, 2], 'min_samples': [5, 10, 15, 20]}
    best_silhouette_score = -1
    
    for params in ParameterGrid(param_grid):
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(X_scaled)
        labels = dbscan.labels_
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_params = params
                best_labels = labels
    
    print(f"Best Parameters: {best_params}, Silhouette Score: {best_silhouette_score}")
    rand_index = adjusted_rand_score(ground_truth, best_labels)
    print(f"Adjusted Rand Index: {rand_index}")
    plot_with_tsne(X_scaled, best_labels, title=f'DBSCAN Clustering with t-SNE ({filename})')

    return best_params, best_silhouette_score, rand_index


print("Processing 'cho.txt' Dataset")
cho_params, cho_silhouette, cho_rand = preprocess_and_cluster_dbscan('cho.txt')

print("\nProcessing 'iyer.txt' Dataset")
iyer_params, iyer_silhouette, iyer_rand = preprocess_and_cluster_dbscan('iyer.txt')




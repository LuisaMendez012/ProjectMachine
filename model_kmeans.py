import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from model import load_dataset


def _build_matrix_for_clustering(rows):
    # Use a subset of features suitable for clustering (temporal + weather + temp/hum/wind)
    X = []
    for row in rows:
        X.append([
            float(row['hr']),
            float(row['weathersit']),
            float(row['season']),
            float(row['temp']),
            float(row['hum']),
            float(row['windspeed']),
        ])
    return np.array(X, dtype=float)


def run_kmeans(n_clusters=4, random_state=42, save_path=None):
    rows = load_dataset()
    X = _build_matrix_for_clustering(rows)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(Xs)
    centers = kmeans.cluster_centers_

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=random_state)
    proj = pca.fit_transform(Xs)
    centers_proj = pca.transform(centers)

    # Save plot
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'kmeans_clusters.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
    plt.scatter(centers_proj[:, 0], centers_proj[:, 1], c='black', s=100, marker='X')
    plt.title(f'K-Means clustering (k={n_clusters}) — PCA projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # Build cluster explanations: approximate centroid in original feature space
    centers_orig = scaler.inverse_transform(centers)
    explanations = []
    feature_names = ['hour', 'weathersit', 'season', 'temp', 'hum', 'windspeed']
    for i, c in enumerate(centers_orig):
        desc = {feature_names[j]: float(round(c[j], 2)) for j in range(len(feature_names))}
        explanations.append({'cluster': int(i), 'centroid': desc})

    # Sample counts per cluster
    unique, counts = np.unique(labels, return_counts=True)
    counts_map = {int(u): int(c) for u, c in zip(unique, counts)}

    # Evaluation metrics for clustering
    inertia = float(kmeans.inertia_)
    silhouette = float(silhouette_score(Xs, labels)) if n_clusters > 1 else None

    # image_path relative to static folder for templates
    rel_path = os.path.join('static', 'images', 'kmeans_clusters.png')

    return {
        'labels': labels.tolist(),
        'centers': centers.tolist(),
        'explanations': explanations,
        'counts': counts_map,
        'image_path': rel_path,
        'inertia': inertia,
        'silhouette': silhouette,
    }


def plot_cluster_counts(counts, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'kmeans_counts.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = [f'Cluster {int(k)}' for k in sorted(counts.keys())]
    values = [counts[k] for k in sorted(counts.keys())]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title('K-Means Cluster Sample Counts')
    plt.ylabel('Number of samples')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return os.path.join('static', 'images', 'kmeans_counts.png')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # fallback; paper uses cluster-based
from sklearn.cluster import KMeans

def z_score_normalization(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def cluster_based_oversampling(X, y, k=5, random_state=42):
    from collections import Counter
    minority_class = 1 if Counter(y)[1] < Counter(y)[0] else 0
    X_min = X[y == minority_class]
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_min)
    
    synthetic_samples = []
    for c in range(k):
        cluster_idx = np.where(clusters == c)[0]
        if len(cluster_idx) < 2:
            continue
        centroid = np.mean(X_min[cluster_idx], axis=0)
        cov = np.cov(X_min[cluster_idx].T) + np.eye(X_min.shape[1]) * 1e-6
        n_synth = max(0, len(cluster_idx) // 2)
        synth = np.random.multivariate_normal(centroid, cov, n_synth)
        synthetic_samples.append(synth)
    
    if synthetic_samples:
        X_synth = np.vstack(synthetic_samples)
        y_synth = np.full(len(X_synth), minority_class)
        X = np.vstack([X, X_synth])
        y = np.hstack([y, y_synth])
    
    return X, y

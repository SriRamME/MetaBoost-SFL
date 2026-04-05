import numpy as np
from sklearn.decomposition import KernelPCA

def apply_kpca(X, n_components=8, kernel='rbf', gamma=0.1):
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma,
                     fit_inverse_transform=True, random_state=42)
    X_kpca = kpca.fit_transform(X)
    print(f"KPCA reduced features from {X.shape[1]} to {X_kpca.shape[1]}")
    return X_kpca, kpca

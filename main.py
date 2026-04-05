import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import z_score_normalization, cluster_based_oversampling
from kpca import apply_kpca
from sfla_optimizer import SFLAOptimizer
from metboost_sfl import MetaBoostSFL
from edge_fog_simulation import simulate_edge_fog_inference
from shap_analysis import explain_model
import os

df = pd.read_csv('datasets/heart_cleveland.csv')  # adjust filename
X = df.drop('target', axis=1).values
y = df['target'].values
feature_names = df.drop('target', axis=1).columns.tolist()

X_norm, scaler = z_score_normalization(X)
X_res, y_res = cluster_based_oversampling(X_norm, y)

X_kpca, kpca_model = apply_kpca(X_res, n_components=8)

X_train, X_test, y_train, y_test = train_test_split(X_kpca, y_res, test_size=0.2, random_state=42, stratify=y_res)

print("Running SFLA optimization...")
optimizer = SFLAOptimizer(X_train, y_train, n_frogs=15, max_iter=20)
best_params = optimizer.optimize()
print("Optimized params:", best_params)

# Train MetaBoost-SFL
model = MetaBoostSFL(best_params)
model.fit(X_train, y_train)

metrics = model.evaluate(X_test, y_test)
print("=== MetaBoost-SFL Performance ===")
for k, v in metrics.items():
    print(f"{k.capitalize()}: {v*100:.2f}%")

# Edge-Fog simulation
simulate_edge_fog_inference(model, X_test[:50])

# SHAP
os.makedirs('results', exist_ok=True)
explain_model(model, X_train, feature_names[:8])  # adjust names after KPCA

# Save model
model.save()

print("Training completed. Results in /results folder.")

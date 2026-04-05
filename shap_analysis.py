import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, feature_names):
    explainer = shap.TreeExplainer(model.model) if hasattr(model.model, 'estimators_') else shap.KernelExplainer(model.predict, X_train[:100])
    shap_values = explainer.shap_values(X_train[:100])
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train[:100], feature_names=feature_names, show=False)
    plt.savefig('results/shap_summary.png')
    plt.close()
    print("SHAP summary plot saved.")

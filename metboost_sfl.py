from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shap
import joblib

class MetaBoostSFL:
    def __init__(self, optimized_params=None):
        if optimized_params is None:
            optimized_params = {'n_estimators': 100, 'learning_rate': 1.0}
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            **optimized_params,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'sensitivity': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auroc': roc_auc_score(y_test, y_prob),
            'specificity': confusion_matrix(y_test, y_pred)[0,0] / (confusion_matrix(y_test, y_pred)[0,0] + confusion_matrix(y_test, y_pred)[0,1])
        }
        return metrics

    def save(self, path='metboost_sfl_model.pkl'):
        joblib.dump(self.model, path)

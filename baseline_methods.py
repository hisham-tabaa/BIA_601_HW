# baseline_methods.py
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_baselines(X_train, y_train, X_test, y_test, k=10):
    results = {}

    # ANOVA F-test (works with standardized data)
    k_use = min(k, X_train.shape[1]) if k > 0 else 1
    anova_selector = SelectKBest(f_classif, k=k_use)
    X_anova = anova_selector.fit_transform(X_train, y_train)
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_anova, y_train)
    X_test_anova = anova_selector.transform(X_test)
    acc_anova = accuracy_score(y_test, model.predict(X_test_anova))
    results['ANOVA F-test'] = {'accuracy': round(acc_anova, 4), 'n_features': k_use}

    # Lasso
    lasso = LassoCV(cv=3, random_state=42, max_iter=5000)
    lasso.fit(X_train, y_train)
    selector = SelectFromModel(lasso, prefit=True)
    X_lasso = selector.transform(X_train)
    if X_lasso.shape[1] == 0:
        results['Lasso'] = {'accuracy': 0.0, 'n_features': 0}
    else:
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_lasso, y_train)
        X_test_lasso = selector.transform(X_test)
        acc_lasso = accuracy_score(y_test, model.predict(X_test_lasso))
        results['Lasso'] = {'accuracy': round(acc_lasso, 4), 'n_features': X_lasso.shape[1]}

    # PCA
    pca_k = min(k_use, X_train.shape[1])
    pca = PCA(n_components=pca_k)
    X_pca = pca.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_pca, y_train)
    X_test_pca = pca.transform(X_test)
    acc_pca = accuracy_score(y_test, model.predict(X_test_pca))
    results['PCA'] = {'accuracy': round(acc_pca, 4), 'n_features': pca_k}

    return results

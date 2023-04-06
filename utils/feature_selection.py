from mlxtend.plotting import plot_pca_correlation_graph
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from mlxtend.feature_extraction import RBFKernelPCA as KPCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from shap import TreeExplainer
import shap
import graphviz
import matplotlib.pyplot as plt
from yellowbrick.target import ClassBalance
import numpy as np


def yellowbrick_class_balance(y, classes=["benigno", "maligno"]):
    visualizer = ClassBalance(labels=classes)

    visualizer.fit(y_train=np.array(y), y_test=None)        # Fit the data to the visualizer
    visualizer.show()   


def plot_pca(X, y, title="First 2 principal components after PCA"):
    plt.scatter(X[y==0, 0], X[y==0, 1], 
                color='red', marker='o', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], 
                color='blue', marker='o', alpha=0.5)
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.title(title)


def create_pca(X, y, n_components=2):
    pca = PCA(n_components=n_components, whitening=True)
    X_pca = pca.fit(X).transform(X)
    plot_pca(X_pca, y)


def create_kpca(X, y, n_components=2, gamma=16.0):
    kpca = KPCA(n_components=n_components, gamma=gamma)
    kpca.fit(X)
    kpca_X = kpca.X_projected_
    plot_pca(kpca_X, y, title="First 2 principal components after RBF Kernel PCA")


def plot_xgboost__feature_importances(X, y):
    plt.figure(figsize=(10,10))
    xgb = XGBClassifier()
    xgb.fit(X, y)
    plt.barh(X.columns, xgb.feature_importances_)


def tree_shap_feature_importances(X_train, y_train, X_test, n_estimators=500, max_depth=6):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    tree_graph = export_graphviz(model, out_file=None, feature_names=X_train.columns, rounded=True, filled=True)
    graphviz.Source(tree_graph)
    shap.initjs()
    explainer = TreeExplainer(model, algorithm="auto", n_jobs=-1)
    shap_values = explainer.shap_values(X_test)
    shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, matplotlib=True)


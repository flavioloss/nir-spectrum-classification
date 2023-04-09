from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np

def model_error(model, X, y):
    kfold = KFold(n_splits=5, shuffle=True)
    score = cross_val_score(model, X, y, scoring='f1', cv=kfold)
    return np.mean(score)


def test_models(X, y):
    rfc = RandomForestClassifier()
    print("RandomFOrest:", model_error(rfc, X, y))

    knn = KNeighborsClassifier()
    print("KNN:", model_error(knn, X, y))

    xgb = XGBClassifier()
    print("XGB:", model_error(xgb, X, y))

    catb = CatBoostClassifier(verbose=0)
    print("CatBoost:", model_error(catb, X, y))

    lgbm = LGBMClassifier()
    print("LightGBM:", model_error(lgbm, X, y))


def simple_catboost():
    catb = CatBoostClassifier(verbose=0, iterations=100)
    return catb

def simple_random_forest():
    rfc = RandomForestClassifier(n_estimators=10)
    return rfc


def fit_simple_catboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                        random_state=1066, stratify=y)

    catb = simple_catboost()
    catb.fit(X_train, y_train)
    pred_cat = catb.predict(X_test)
    print('acc:', round(accuracy_score(y_test, pred_cat), 3))
    print('fi-score:', round(f1_score(y_test, pred_cat), 3))
    print('precision:', round(precision_score(y_test, pred_cat), 3))
    print('recall:', round(recall_score(y_test, pred_cat), 3))
    return catb
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import numpy as np
import pandas as pd

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
data = pd.read_csv('../../Datasets/high_diamond_ranked_10min.csv')
data.head()
# -

# # Feature Selection

# +
from sklearn.feature_selection import SelectKBest, f_classif

# Choose target variable
y = data['blueWins']
X = data
X.drop(columns=['blueWins'], inplace=True)

def get_data(features=[]):
    if len(features) > 0:
        return data[features]
    else:
        # choose features
        selector = SelectKBest(f_classif, k)
        
        X_new = selector.fit_transform(X, y)
        
        # Get back the features we've kept, zero out all other features
        selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                        index=X.index,
                                        columns=X.columns)
        
        # Dropped columns have values of all zero, so var is zero, drop them
        selected_columns = selected_features[selected_features.var() != 0]
        
        return selected_features[selected_columns]
    
X = get_data(['blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood', 'blueKills', 'blueAssists', 'blueEliteMonsters', 'blueDragons', 'blueHeralds', 'blueTotalMinionsKilled', 'blueTowersDestroyed', 'blueCSPerMin', 'blueAvgLevel', 'blueTotalExperience'])
X
# -

# # Random Forest

# +
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Train and prediction
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf.fit(X, y)

# Cross validation
cross_val_score(rf, X, y, cv=5).mean()
# -

# # Naive Bayes

# +
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB();
nb.fit(X, y)

# Cross-validation
cross_val_score(nb, X, y, cv=5).mean()
# -

# # Support Vector Machine

# +
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

svc = make_pipeline(MinMaxScaler(), SVC(kernel='rbf', C=1000))
svc.fit(X, y)

cross_val_score(svc, X, y, cv=5).mean()
# -

# # Extreme Gradient Boosting

# +
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.01, n_estimators=1000)
xgb.fit(X, y)

cross_val_score(xgb, X, y, cv=5).mean()
# -

# # Light Gradient Boosting

# +
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(learning_rate=0.01, n_estimators=1000)
lgbm.fit(X, y)

cross_val_score(lgbm, X, y, cv=5).mean()
# -



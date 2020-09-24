# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import numpy as np
import pandas as pd

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
train = pd.read_csv('../../Datasets/titanic_train.csv', index_col=0)
train.head()
# -

test = pd.read_csv('../../Datasets/titanic_test.csv', index_col=0)
test.head()

# Calculate the percentage of women who survived

surviving_women = train[train['Sex'] == 'female']['Survived']
sum(surviving_women) / len(surviving_women)

# Calculate the percentage of men who survived

surviving_men = train[train['Sex'] == 'male']['Survived']
sum(surviving_men) / len(surviving_men)

# # Feature Engineering
#
# ## Select target variables and features

# +
# Choose target variable
y = train['Survived']

# Select features
features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']
X = train[features]
X_test = test[features]
# -

# ## Missing Values

# +
# In order to use 'Age' as a feature we need to impute missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imputer.fit_transform(X))
X_test_imp = pd.DataFrame(imputer.transform(X_test))

X_imp.columns = X.columns
X_test_imp.columns = X_test.columns

X = X_imp
X_test = X_test_imp
# -

X

X_test

# ## Scaling

# +
from sklearn import preprocessing

preprocessing.scale(X['Fare'], copy=False)
preprocessing.scale(X['Age'], copy=False)
preprocessing.scale(X_test['Fare'], copy=False)
preprocessing.scale(X_test['Age'], copy=False)
X
# -

X_test

# ## Categorical variables

# +
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Label encoding
categorical = ['Sex', 'Embarked']

imputer = SimpleImputer(strategy='most_frequent')

imp_cols_train = pd.DataFrame(imputer.fit_transform(train[categorical]))
imp_cols_test = pd.DataFrame(imputer.transform(test[categorical]))

imp_cols_train.columns = categorical
imp_cols_test.columns = categorical

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imp_cols_train))
OH_cols_test = pd.DataFrame(OH_encoder.transform(imp_cols_test))

OH_cols_train.index = imp_cols_train.index
OH_cols_test.index = imp_cols_test.index

for col in OH_cols_train.columns:
    X[col] = OH_cols_train[col]
    X_test[col] = OH_cols_test[col]
    
X.head()
# -

# ## Feature Selection

# +
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)

X_new = selector.fit_transform(X, y)

# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                index=X.index,
                                columns=X.columns)

# Dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0]
X = selected_features[selected_columns]

# Get the test dataset with the selected features
X_test = X_test[selected_columns]

X
# -

# # Random Forest

# ## Learning Curves

# +
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import learning_curve

import seaborn as sns
sns.set()

estimators = [1, 10, 100, 1000]

fig = plt.figure(constrained_layout=True, figsize=(10, 6))
grid_spec = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

for row in range(2):
    for col in range(2):
        n = estimators[2 * row + col]
        N, train_lc, val_lc = learning_curve(RandomForestClassifier(n_estimators=n, max_depth=5, random_state=1),
                                             X, y, cv=7)

        ax = fig.add_subplot(grid_spec[row, col])

        ax.plot(N, np.mean(train_lc, 1), color='blue', marker='.', label='training_score')
        ax.plot(N, np.mean(val_lc, 1), color='red', marker='.', label='validation_score')
        ax.hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray', linestyle='dashed')

        ax.set_xlabel('training size')
        ax.set_ylabel('score')
        ax.set_title(f'Estimators = {n}', size=14)
        ax.legend(loc='best')
# -

# ## Grid Search

# +
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [1, 10, 100, 1000],
              'max_depth': [5, 10, 15, 20]}

grid = GridSearchCV(RandomForestClassifier(random_state=1), param_grid)
grid.fit(X, y)
print(grid.best_params_)
# -

# ## Training and Prediction

# +
from sklearn.model_selection import cross_val_score

rf = grid.best_estimator_
rf.fit(X, y)

# Cross-validation
cross_val_score(rf, X, y, cv=5).mean()
# -

# # Support Vector Machine

# ## Grid Search

# +
from sklearn.svm import SVC

param_grid = {'C': [10, 100, 1000, 2000],
             'kernel': ['rbf', 'sigmoid']}

grid = GridSearchCV(SVC(), param_grid)
grid.fit(X, y)
grid.best_params_
# -

# ## Training and prediction

# +
svc = grid.best_estimator_
svc.fit(X, y)

cross_val_score(svc, X, y, cv=5).mean()
# -

# # Extreme Gradient Boosting

# +
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.1)
xgb.fit(X, y)

cross_val_score(xgb, X, y, cv=5).mean()
# -

# # Light Gradient Boosting

# +
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(learning_rate=0.1)
lgbm.fit(X, y)

cross_val_score(lgbm, X, y, cv=5).mean()
# -

# # Generate output

# +
predictions = rf.predict(X_test)

output = pd.DataFrame({'PassengerId': test.index, 'Survived': predictions})
output.to_csv('output.csv', index=False)

print('Done')

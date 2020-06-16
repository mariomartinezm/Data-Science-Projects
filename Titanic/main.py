# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: 'Python 3.7.6 64-bit (''venv'': venv)'
#     language: python
#     name: python37664bitvenvvenv8099a62852874eaebba06d1f38f49ab5
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

selector = SelectKBest(f_classif, k=6)

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

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Train and prediction
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf.fit(X, y)

# Cross-validation
cross_val_score(rf, X, y, cv=5).mean()
# -

# # Support Vector Machine

# +
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svc = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1000))
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
# -



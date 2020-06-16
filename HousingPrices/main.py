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
train = pd.read_csv('../../Datasets/housing_prices_train.csv', index_col=0)
train.head(10)
# -

test = pd.read_csv('../../Datasets/housing_prices_test.csv', index_col=0)
test.head(10)

# # Select target variable and features

# +
y = train['SalePrice']

# Select non categorical features and exclude target variable
X = train.select_dtypes(exclude=['object'])
X.drop(columns=['SalePrice'], inplace=True)

X_test = test.select_dtypes(exclude=['object'])
# -

# # Feature Engineering

# ## Missing values

# +
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

# Number of missing values in each column of training data
missing_val_count_by_column = (X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

missing_val_count_by_column = (X_test.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

X_imp = pd.DataFrame(imputer.fit_transform(X))
X_test_imp = pd.DataFrame(imputer.transform(X_test))

X_imp.columns = X.columns
X_test_imp.columns = X_test.columns

X = X_imp
X_test = X_test_imp
# -

# ## Label encoding

# +
from sklearn.preprocessing import LabelEncoder
features = ['BldgType', 'CentralAir', 'Heating', 'HeatingQC', 'SaleCondition', 'ExterCond', 'ExterQual', 'Foundation', 'HouseStyle', 'LandContour', 'LotShape', 'RoofStyle', 'RoofMatl', 'Street', 'Neighborhood']

# Get list of categorical features
categorical = [col for col in features if train[col].dtype == 'O']
print(categorical)

label_encoder = LabelEncoder()

for col in features:
    X[col] = label_encoder.fit_transform(train[col])
    X_test[col] = label_encoder.transform(test[col])
    
X.head()
# -

# ## One Hot Encoding

# +
from sklearn.preprocessing import OneHotEncoder

features = ['MSZoning', 'Utilities', 'KitchenQual', 'Fence', 'GarageType']

imputer = SimpleImputer(strategy='most_frequent')

imp_cols_train = pd.DataFrame(imputer.fit_transform(train[features]))
imp_cols_test = pd.DataFrame(imputer.transform(test[features]))

imp_cols_train.columns = features
imp_cols_test.columsn = features

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

# # Random Forest

# +
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=1)
rf.fit(X, y)

-1 * cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
# -

# # Support Vector Machines

# +
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1000))
svr.fit(X, y)

-1 * cross_val_score(svr, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
# -

# # Extreme Gradient Boosting

# +
from xgboost import XGBRegressor

xgb = XGBRegressor(learning_rate=0.1)
xgb.fit(X, y)

-1 * cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
# -

# # Light Gradient Boosting

# +
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(learning_rate=0.1, n_estimators=1000)
lgbm.fit(X, y)

-1 * cross_val_score(lgbm, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
# -

# # Grid search

# +
from sklearn.model_selection import GridSearchCV

param_grid = {'learning_rate': np.linspace(0.01, 1, 5),
              'n_estimators': [100, 500, 1000]}

grid = GridSearchCV(XGBRegressor(), param_grid, cv=5)
grid.fit(X, y)

print(grid.best_params_)

xgb = grid.best_estimator_

-1 * cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
# -

# # Prepare Output

# +
predictions = xgb.predict(X_test)

output = pd.DataFrame({'Id': test.index, 'SalePrice': predictions})
output.to_csv('output.csv', index=False)

print('Done')
# -



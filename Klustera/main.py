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
# -

# # Exploratory Data Analysis
#
# The dataset comprises the following attributes:
#
#   * **Device_mac** is a unique cellphone identifier, we assume each cell corresponds to a single person.
#   * **Branch_office** is the store of a particular client.
#   * **visitor** is an indicator variable that specifies wether a person is a visitor or not.
#   * **tiempodeses** is the elapsed time of a session in seconds.
#   * **device_mac** is the MAC address of the cellphone.

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
train = pd.read_csv('../../Datasets/e.csv', index_col=0)
train.head()
# -

test = pd.read_csv('../../Datasets/v.csv', index_col=0)
test.head()

# # Feature Selection

# +
# Choose target variable and features
y = train['visitor']
features = ['tiempodeses']

X = train[features]
X_test = test[features]

X.head()
# -

# # Feature Engineering

# +
from sklearn.preprocessing import LabelEncoder

#categorical = [col for col in train.columns if train[col].dtype == 'O']
categorical = ['day_of_week_tz']

X_cat = train[categorical]
X_test_cat = test[categorical]

le = LabelEncoder()
for col in categorical:
    le.fit(X_cat[col])
    X[col] = le.transform(X_cat[col])
    X_test[col] = le.transform(X_test_cat[col])
    
X_test
# -

# # Random Forest

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier()
rf.fit(X, y)

print(cross_val_score(rf, X, y, cv=5).mean())
# -

# # Extreme Gradient Boosting

# +
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.1)
xgb.fit(X, y)
cross_val_score(xgb, X, y, cv=5).mean()
# -

# # Prepare Output

# +
y_test = xgb.predict(X_test)
test['visitor'] = y_test
test.to_csv('output.csv', index=False)

print('Done')
# -



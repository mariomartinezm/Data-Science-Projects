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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
train = pd.read_csv('../../Datasets/shelter-animal-outcomes/train.csv.gz')
train.head(10)
# -

test = pd.read_csv('../../Datasets/shelter-animal-outcomes/test.csv.gz', index_col=0)
test.head(10)

# # Exploratory data analysis

# Get columns with missing values:

print([col for col in train.columns if train[col].isna().any()])
print([col for col in test.columns if test[col].isna().any()])

# Get the total number of outcome categories:

train['OutcomeType'].value_counts()

# Get the number of sex categories

train['SexuponOutcome'].value_counts()

# Get the number of 'AnimalType' categories:

train['AnimalType'].value_counts()

# Get the number of 'Breed' categories:

train['Breed'].value_counts()


# # Feature Engineering

# ## Data cleaning

# ### Measure age in days

# The column *AgeuponOutcome* contains the age of an animal at the time the corresponding outcome occurred. However the values for this column contains strings such as "2 years", "2 months", "3 weeks", etc. In order to use these values for training we need to convert them to a common numeric scale. To fix this, we can convert all values to days as follows:

# +
def extract_age(df, date_part):
    part = df[df['AgeuponOutcome'].str.contains(date_part) & (df['AgeuponOutcome'].isna() is not True)]['AgeuponOutcome']
    
    return part.apply(lambda x: x.split()[0])

age_years = extract_age(train, 'year').astype('int') * 365
age_months = extract_age(train, 'month').astype('int') * 30
age_weeks = extract_age(train, 'week').astype('int') * 7
age_days = extract_age(train, 'day').astype('int')

train.loc[age_years.index, 'AgeuponOutcome'] = age_years
train.loc[age_months.index, 'AgeuponOutcome'] = age_months
train.loc[age_weeks.index, 'AgeuponOutcome'] = age_weeks
train.loc[age_days.index, 'AgeuponOutcome'] = age_days

train['AgeuponOutcome'] = train['AgeuponOutcome'].astype('Int64')
train['AgeuponOutcome']
# -

# Now we do the same with the test dataset:

# +
age_years = extract_age(test, 'year').astype('int') * 365
age_months = extract_age(test, 'month').astype('int') * 30
age_weeks = extract_age(test, 'week').astype('int') * 7
age_days = extract_age(test, 'day').astype('int')

test.loc[age_years.index, 'AgeuponOutcome'] = age_years
test.loc[age_months.index, 'AgeuponOutcome'] = age_months
test.loc[age_weeks.index, 'AgeuponOutcome'] = age_weeks
test.loc[age_days.index, 'AgeuponOutcome'] = age_days

test['AgeuponOutcome']
# -

# ## Missing Values

# +
from sklearn.impute import SimpleImputer

cols_missing = ['SexuponOutcome']

si = SimpleImputer(strategy='most_frequent')

imp_cols_train = pd.DataFrame(si.fit_transform(train[cols_missing]))
imp_cols_train.columns = cols_missing

train[cols_missing] = imp_cols_train

si = SimpleImputer(strategy='mean')

imp_cols_train = pd.DataFrame(si.fit_transform(train[['AgeuponOutcome']]))
imp_cols_test = pd.DataFrame(si.transform(test[['AgeuponOutcome']]))

train['AgeuponOutcome'] = imp_cols_train
test['AgeuponOutcome'] = imp_cols_test
# -

# ## One Hot Encoding

# +
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categorical = ['SexuponOutcome', 'AnimalType']

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe_cols_train = pd.DataFrame(ohe.fit_transform(train[categorical]))
ohe_cols_test = pd.DataFrame(ohe.transform(test[categorical]))

ohe_cols_train.index = train.index
ohe_cols_test.index = test.index

for col in ohe_cols_train.columns:
    train[col] = ohe_cols_train[col]
    test[col] = ohe_cols_test[col]
# -

train.head(10)

test.head(10)

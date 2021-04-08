# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import numpy as np
import pandas as pd

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
train = pd.read_csv('../../Datasets/shelter-animal-outcomes/train.csv.gz')
train.head(10)
# -

test = pd.read_csv('../../Datasets/shelter-animal-outcomes/test.csv.gz', index_col=0)
test.head(10)

# # Data cleaning

# Before attempting any kind of analysis it is important to clean the data, e.g., to impute missing values or to perform transformations on the data so they may be used for statistical analysis or traning of a machine learning algorithm.

# ## Missing Values

# Get columns with missing values:

print([col for col in train.columns if train[col].isna().any()])
print([col for col in test.columns if test[col].isna().any()])

# +
from sklearn.impute import SimpleImputer

cols_missing = ['SexuponOutcome']

si = SimpleImputer(strategy='most_frequent')

imp_cols_train = pd.DataFrame(si.fit_transform(train[cols_missing]))
imp_cols_train.columns = cols_missing

train[cols_missing] = imp_cols_train
# -

# Before imputing the *AgeUponOutcome* feature, it is necessary to transform this feature so it represents an age given in days.

# ## Parse dates

train['DateTime'] = pd.to_datetime(train['DateTime'])
train['DateTime']

test['DateTime'] = pd.to_datetime(test['DateTime'])
test['DateTime']


# # Feature Engineering

# ## Measure age in days

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

test['AgeuponOutcome'] = test['AgeuponOutcome'].astype('Int64')
test['AgeuponOutcome']
# -


# Now we can impute the missing values in the *AgeuponOutcome* column:

# +
si = SimpleImputer(strategy='mean')

imp_cols_train = pd.DataFrame(si.fit_transform(train[['AgeuponOutcome']]))
imp_cols_test = pd.DataFrame(si.transform(test[['AgeuponOutcome']]))

train['AgeuponOutcome'] = imp_cols_train
test['AgeuponOutcome'] = imp_cols_test
# -

# ## Month of outcome

train['MonthOutcome'] = [date.month for date in train['DateTime']]
train['MonthOutcome']

test['MonthOutcome'] = [date.month for date in test['DateTime']]
test['MonthOutcome']

# # Year of outcome

train['YearOutcome'] = [date.year for date in train['DateTime']]
train['YearOutcome']

test['YearOutcome'] = [date.year for date in test['DateTime']]
test['YearOutcome']

# ## One Hot Encoding

# +
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categorical = ['SexuponOutcome', 'AnimalType']

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

ohe_cols_train = pd.DataFrame(ohe.fit_transform(train[categorical]))
ohe_cols_train.columns = ohe.get_feature_names(['SexuponOutcome', 'AnimalType'])

ohe_cols_test = pd.DataFrame(ohe.transform(test[categorical]))
ohe_cols_test.columns = ohe.get_feature_names(['SexuponOutcome', 'AnimalType'])

ohe_cols_train.index = train.index
ohe_cols_test.index = test.index

for col in ohe_cols_train.columns:
    train[col] = ohe_cols_train[col]
    test[col] = ohe_cols_test[col]
# -

train.head(10)

test.head(10)

# # Exploratory data analysis

# The data contains several categorical variables (*SexUponOutcome*, *Breed*, *Color*, etc), the corresponding frequencies can be obtained as follows:

# +
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

features = ['OutcomeType', 'SexuponOutcome', 'AnimalType']

fig, ax = plt.subplots(1, len(features))

fig.set_figheight(5)
fig.set_figwidth(15)

for i, feature in enumerate(train[features]):
    train[feature].value_counts().plot(kind='bar', ax=ax[i]).set_title(feature)
# -

# After transforming the column *AgeuponOutcome* we can obtain its distribution as follows:

# +
fig, ax = plt.subplots(1, 2)

fig.set_figheight(5)
fig.set_figwidth(15)

sns.distplot(train['AgeuponOutcome'] / 365, bins=20, kde=True, ax=ax[0])
sns.distplot(test['AgeuponOutcome'] / 365, bins=20, kde=True, ax=ax[1])
# -

# From these plots we can conclude that most animals in the shelter (most of them dogs) are quite young, at most 5 years old. Fortunately, only a small portion of the animals died or were sacrificed during their stance at the shelter. In order to understand the conditions that lead to an animal not being adopted it is critical to investigate the demographics of the populations, e.g., the percentage of males and females that were adopted.

# +
adopted = train[train['OutcomeType'] == 'Adoption']

males_adopted = train[(train['OutcomeType'] == 'Adoption') & (train['SexuponOutcome'].str.contains('Male'))]
females_adopted = train[(train['OutcomeType'] == 'Adoption') & (train['SexuponOutcome'].str.contains('Female'))]

total_adopted = len(adopted)
print('Percentage of males adopted = {}'.format(round(len(males_adopted) / total_adopted * 100, 2)))
print('Percentage of females adopted = {}'.format(round(len(females_adopted) / total_adopted * 100, 2)))

fig = plt.figure()

plt.title('Adoption by sex')

plt.pie([len(males_adopted), len(females_adopted)],
        labels=['Males adopted', 'Females adopted'],
        autopct='%1.2f%%')

plt.plot()

# +
males = train[train['SexuponOutcome'].str.contains('Male')]
males_dead = train[train['SexuponOutcome'].str.contains('Male') & train['OutcomeType'].isin(['Euthanasia', 'Died'])]
print('Percentage of males dead: {}'.format(len(males_dead) / len(males)))

females = train[train['SexuponOutcome'].str.contains('Female')]
females_dead = train[train['SexuponOutcome'].str.contains('Female') & train['OutcomeType'].isin(['Euthanasia', 'Died'])]
print('Percentage of females dead: {}'.format(len(females_dead) / len(females)))

train['OutcomeType'].isin(['Euthanasia', 'Died'])
# -

# So only a small proportion of animals died, with males having a slightly greater death rate.

train['Breed'].value_counts()

# ## Outcomes per month
#
# To observe the number of samples per month, for the train dataset, we proceed as follows:

train['MonthOutcome'].value_counts().plot(kind='bar').set_title('Month Outcome - Train')

# The plot shows that there are more outcomes in winter months than every other season. However we should investigate how many adoptions and deaths occur in a monthly basis.

# +
fig, ax = plt.subplots(1, len(np.unique(train['OutcomeType'])))

fig.set_figheight(5)
fig.set_figwidth(15)

for i, val in enumerate(np.unique(train['OutcomeType'])):
    df = train[train['OutcomeType'] == val]
    df['MonthOutcome'].value_counts().plot(kind='bar', ax=ax[i]).set_title(val)
# -

# # Feature Selection

# +
y = train['OutcomeType']

features = ['AgeuponOutcome']
features.extend(ohe.get_feature_names(['SexuponOutcome', 'AnimalType']))

X_train = train[features]
X_test = test[features]

X_train.head(10)
# -


# # Random Forest

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier()
rf.fit(X_train, y)

cross_val_score(rf, X_train, y, cv=5).mean()
# -



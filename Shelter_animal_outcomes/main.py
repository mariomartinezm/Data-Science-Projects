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
train
# -

test = pd.read_csv('../../Datasets/shelter-animal-outcomes/test.csv.gz', index_col=0)
test

# # Exploratory data analysis

# Get columns with missing values:

[col for col in train.columns if train[col].isna().any()]

# Get the total number of outcome categories:

train['OutcomeType'].value_counts()

# Get the number of sex categories

train['SexuponOutcome'].value_counts()

# Get the number of 'AnimalType' categories:

train['AnimalType'].value_counts()

# Get the number of 'Breed' categories:

train['Breed'].value_counts()

# # Target variable and features selection



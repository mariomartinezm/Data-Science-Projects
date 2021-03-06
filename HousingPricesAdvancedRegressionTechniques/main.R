# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

library(ggplot2)
library(modelr)
library(ranger)

# # Housing Prices Competition
#
# ## Introduction
#
# This is one of Kaggle's competitions aimed at beginners. The objective is to predict the final sale price of a house using as much as 79 explanatory variables that contain information such as the number of bedrooms, the quality of plumbing, etc.
#
# We start by loading the training and test data:

train <- read.csv("D:/Documents/Python/DSandML/Datasets/house-prices-advanced-regression-techniques/train.csv", stringsAsFactors=TRUE)
head(train)

test <- read.csv("D:/Documents/Python/DSandML/Datasets/house-prices-advanced-regression-techniques/test.csv", stringsAsFactors=TRUE)
head(test)

# After inspecting the data it is easy to see that our target variable is "Saleprice". It is also convenient to get a summary of the data to get some important statistics and determine which features contain missing values:

summary(train)

summary(test)

# Since there are categorical features that are incorrectly read as numeric, it is necessary to transform the data:

train <- transform(train, MSSubClass=as.factor(MSSubClass))
test <- transform(test, MSSubClass=as.factor(MSSubClass))

# ## Data Cleaning

# Select all numeric features:

nums <- unlist(lapply(train, is.numeric))
num_train = train[, nums]
num_train$Id <- NULL
head(num_train)

nums <- unlist(lapply(test, is.numeric))
num_test <- test[, nums]
num_test$Id <- NULL
head(num_test)

# ### Missing Values

imputeNA <- function(data, strategy)
{
    val <- NA
    if(strategy == "median")
    {
        val <- median(data, na.rm=TRUE)
        data <- ifelse(is.na(data), val, data)
    }
    else if(strategy == "mean")
    {
        val <- mean(data, na.rm=TRUE)
        data <- ifelse(is.na(data), val, data)
    }
    
    data
}

# Impute train data:

# +
for(col in colnames(num_train))
{
    if(any(is.na(num_train[, col])))
    {
        num_train[, col] <- imputeNA(num_train[, col], "median")
    }
}

any(is.na(num_train$LotFrontage))
# -

for(col in colnames(num_test))
{
    if(any(is.na(num_test[, col])))
    {
        num_test[, col] <- imputeNA(num_test[, col], "median")
    }
}

# ### Add numerical features to train and test data

# +
numerical = c("LotFrontage", "LotArea", "YearBuilt", "MasVnrArea",
              "GarageCars", "BsmtFinSF1", "X1stFlrSF", "X2ndFlrSF",
              "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
              "WoodDeckSF", "BsmtUnfSF", "YearRemodAdd", "BsmtFullBath",
              "OverallCond", "FullBath", "PoolArea", "MiscVal")

X_train <- num_train[, numerical]
X_train$SalePrice <- num_train[, c("SalePrice")]
head(X_train)
# -

X_test <- num_test[, numerical]
head(X_test)

# ### MSZoning

# This variable identifies the general zoning classification of the sale. The *test* dataset contains some missing values for this variable, however, we can't simply impute using some statistic such as the global mode. Instead, it is better to impute using the mode of all rows having the same neighborhood as the rows having the missing values.
#
# First, it is neccesary to get the index of missing values from the **MSZoning** variable:

which(is.na(test['MSZoning']))

# Since there are only 4 missing values, we can inspect every row to determine the **Neighborhood** of every row with missing data.

test[c(456, 757, 791, 1445), "Neighborhood"]

# The first three rows belong to the IDOTRR neighborhood, the last belongs to Mitchell.
# Let's identify the most common value for the feature **MSZoning** when **Neighborhood** takes the value IDOTRR.

test_idotrr <- test[test$Neighborhood == "IDOTRR", c("MSZoning")]
summary(test_idotrr)

# Finally, let's impute the data:

test[c(456, 757, 791), "MSZoning"] <- "RM"

# Let's repeat the process for the Mitchel neighborhood: 

test_mitchel <- test[test$Neighborhood == "Mitchel", c("MSZoning")]
summary(test_mitchel)

test[c(1445), "MSZoning"] <- "RL"
summary(test$MSZoning)

# ### Electrical
#
# There is one row in the *training* data set with a missing value:

summary(train$Electrical)

# Let's determine the index where the missing value occurs:

which(is.na(train["Electrical"]))

# Finally let's impute it with the mode according to neighborhood:

train[1380, c("Neighborhood", "Electrical")]

train_timber <- train[train$Neighborhood == "Timber", c("Electrical")]
summary(train_timber)

train[1380, "Electrical"] <- "SBrkr"
summary(train$Electrical)

# ### GarageQual
#
# This categorical feature uses the string "NA" to represent one of its levels. To stop R from misinterpreting this value it is necessary to replace all occurences of "NA".

train$GarageQual <- as.character(train$GarageQual)
indices <- which(is.na(train$GarageQual))
train$GarageQual[indices] <- "NoGarage"
train$GarageQual = as.factor(train$GarageQual)
summary(train$GarageQual)

test$GarageQual <- as.character(test$GarageQual)
indices <- which(is.na(test$GarageQual))
test$GarageQual[indices] <- "NoGarage"
test$GarageQual = as.factor(test$GarageQual)
summary(test$GarageQual)

# ### FireplaceQu
#
# Same case as **GarageQual**.

train$FireplaceQu <- as.character(train$FireplaceQu)
indices <- which(is.na(train$FireplaceQu))
train$FireplaceQu[indices] <- "NoKitchen"
train$FireplaceQu = as.factor(train$FireplaceQu)
summary(train$FireplaceQu)

test$FireplaceQu <- as.character(test$FireplaceQu)
indices <- which(is.na(test$FireplaceQu))
test$FireplaceQu[indices] <- "NoKitchen"
test$FireplaceQu = as.factor(test$FireplaceQu)
summary(test$FireplaceQu)

# ### Fence
#
# Another categoric variable with a problematic level.

train$Fence <- as.character(train$Fence)
indices <- which(is.na(train$Fence))
train$Fence[indices] <- "NoFence"
train$Fence = as.factor(train$Fence)
summary(train$Fence)

test$Fence <- as.character(test$Fence)
indices <- which(is.na(test$Fence))
test$Fence[indices] <- "NoFence"
test$Fence = as.factor(test$Fence)
summary(test$Fence)

# ### BsmtQual
#
# Same as before.

train$BsmtQual <- as.character(train$BsmtQual)
indices <- which(is.na(train$BsmtQual))
train$BsmtQual[indices] <- "NoBasement"
train$BsmtQual = as.factor(train$BsmtQual)
summary(train$BsmtQual)

test$BsmtQual <- as.character(test$BsmtQual)
indices <- which(is.na(test$BsmtQual))
test$BsmtQual[indices] <- "NoBasement"
test$BsmtQual = as.factor(test$BsmtQual)
summary(test$BsmtQual)

# ### Functional
#
# The *Test* dataset contains two rows with missing values:

which(is.na(test$Functional))

test[c(757, 1014), c("Neighborhood", "Functional")]

ggplot(data=test[test$Neighborhood == "IDOTRR",]) +
    geom_bar(mapping=aes(x=Functional, fill=Functional))

# Most properties in the "IDOTRR" neighborhood are of "Typ" (typical) functionality, so let's impute our rows using such value.

test[c(757, 1014), c("Functional")] <- "Typ"
summary(test$Functional)

# ### SaleType
#
# There is one missing value in the *test* dataset.

which(is.na(test$SaleType))

test[c(1030), c("YrSold", "YearBuilt", "SaleType")]
summary(test$YrSold)

ggplot(data=test) +
    geom_bar(mapping=aes(x=SaleType, fill=SaleType))

test[c(1030), c("SaleType")] <- "WD"
summary(test$SaleType)

# ### Add categorical variables to train and test data:

# +
categorical = c("MSZoning", "Neighborhood", "BldgType", "ExterQual", "ExterCond", "HeatingQC", "GarageQual", "BsmtQual")

for(col in categorical)
{
    X_train[col] = train[col]
    X_test[col] = test[col]
}

head(X_train)
# -

head(X_test)

# ## Linear Regression

# ### Training

model <- lm(SalePrice ~ ., data=X_train)
summary(model)

# ### Predictions

predictions <- predict(model, newdata=X_train)
train_results = data.frame("SalePrice"=X_train$SalePrice, "Predictions"=predictions)
head(train_results)

predictions <- predict(model, newdata=X_test)
test_results = data.frame("SalePrice"=predictions)
head(test_results)

# ### Analizing prediction quality
#
# One way to determine the quality of our predictions is to plot our target variable as a function of our predictions. If the predictions are good the plot will be dots arranged near the line $y = x$, which is called the line of *perfect prediction*.

ggplot(data=train_results, aes(y=SalePrice, x=Predictions)) + geom_point(size=1)

# A similar approach is the *residual plot*, where the predictions errors are plotted as a function of the predictions. In this case, the line of perfect prediction is the line $y = 0$

ggplot(data=train_results, aes(x=Predictions, y=SalePrice - Predictions)) +
    geom_point(alpha=0.2, size=1) +
    geom_smooth(aes(x=Predictions, y=SalePrice - Predictions), color="black")

# Finally we can also use the *mean absolute error* and *mean squared error* to characterize the quality of the predictions:

mae(model, X_train)
mse(model, X_train)

# +
#library(caret)

#set.seed(42)

#train.control <- trainControl(method="cv", number=10)

#model <- train(SalePrice ~ ., data=X_train, method="lm", trControl=train.control)
#model
# -

# ### Generate results file

# +
results <- data.frame(
    "id"=test$Id,
    "SalePrice"=test_results$SalePrice
)

head(results)
# -

write.csv(results, "results_lr.csv", row.names=FALSE)
print("Done")

# ## Random Forest

# ### Training

rf <- ranger(SalePrice ~ ., data=X_train, num.trees=1000, max.depth=0)
rf$prediction.error

# ### Predictions

pred <- predict(rf, data=X_train)
X_train$Predictions <- pred$predictions
head(X_train)

pred <- predict(rf, data=X_test)
X_test$SalePrice <- pred$predictions
head(X_test)

# ### Analyzing Prediction quality

ggplot(data=X_train, aes(y=SalePrice, x=Predictions, color=ExterQual)) + geom_point(size=1)

ggplot(data=X_train, aes(x=Predictions, y=SalePrice - Predictions)) +
    geom_point(alpha=0.2, size=1) +
    geom_smooth(aes(x=Predictions, y=SalePrice - Predictions), color="black")

# ### Generate Results File

# +
results <- data.frame(
    "id"=test$Id,
    "SalePrice"=X_test$SalePrice
)

head(results)
# -

write.csv(results, "results_rf.csv", row.names=FALSE)
print("Done")



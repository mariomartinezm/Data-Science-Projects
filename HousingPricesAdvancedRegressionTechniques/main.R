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
X_train <- train[, nums]
X_train$Id <- NULL
head(X_train)

nums <- unlist(lapply(test, is.numeric))
X_test <- test[, nums]
X_test$Id <- NULL
head(X_test)

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
for(col in colnames(X_train))
{
    if(any(is.na(X_train[, col])))
    {
        X_train[, col] <- imputeNA(X_train[, col], "median")
    }
}

any(is.na(X_train$LotFrontage))
# -

for(col in colnames(X_test))
{
    if(any(is.na(X_test[, col])))
    {
        X_test[, col] <- imputeNA(X_test[, col], "median")
    }
}

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

# ### Add categorical variables to train and test data:

# +
categorical = c("MSZoning", "Neighborhood", "BldgType", "ExterQual", "ExterCond", "HeatingQC", "GarageQual")

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

model <- lm(SalePrice ~ 
            LotFrontage + 
            LotArea + 
            YearBuilt + 
            MasVnrArea + 
            GarageCars +
            BsmtFinSF1 +
            X1stFlrSF +
            X2ndFlrSF +
            BedroomAbvGr +
            KitchenAbvGr +
            TotRmsAbvGrd +
            WoodDeckSF +
            BsmtUnfSF +
            YearRemodAdd +
            BsmtFullBath +
            OverallCond +
            FullBath +
            PoolArea +
            Neighborhood +   # <- Categorical varibles start here
            BldgType +
            ExterQual +
            ExterCond +
            HeatingQC +
            GarageQual,
            data=X_train)
summary(model)

# ### Predictions

X_train$Predictions <- predict(model, newdata=X_train)
head(X_train)

X_test$SalePrice <- predict(model, newdata=X_test)
head(X_test)

# ### Analizing prediction quality
#
# One way to determine the quality of our predictions is to plot our target variable as a function of our predictions. If the predictions are good the plot will be dots arranged near the line $y = x$, which is called the line of *perfect prediction*.

# +
library(ggplot2)

ggplot(data=X_train, aes(y=SalePrice, x=Predictions)) + geom_point(size=1)
# -

# A similar approach is the *residual plot*, where the predictions errors are plotted as a function of the predictions. In this case, the line of perfect prediction is the line $y = 0$

ggplot(data=X_train, aes(x=Predictions, y=SalePrice - Predictions)) +
    geom_point(alpha=0.2, size=1) +
    geom_smooth(aes(x=Predictions, y=SalePrice - Predictions), color="black")

# Finally we can also use the *mean absolute error* to characterize the quality of the predictions:

# +
library(modelr)

mae(model, train)

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
    "SalePrice"=X_test$SalePrice
)

head(results)
# -

write.csv(results, "results_lr.csv", row.names=FALSE)
print("Done")

# ## 

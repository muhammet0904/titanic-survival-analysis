#%%


# Titanic Survival Prediction Project
# This script performs data cleaning, feature engineering, and logistic regression to predict survival on the Titanic dataset.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load datasets
df = pd.read_csv("C:/Users/mami/datacampcsv/projects/titanic/gender_submission.csv")
test = pd.read_csv("C:/Users/mami/datacampcsv/projects/titanic/test.csv")
train = pd.read_csv("C:/Users/mami/datacampcsv/projects/titanic/train.csv")

# Step 2: Copy test data
test_copy = test.copy()

# Step 3: Extract titles from names
test["title"] = test["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
train["title"] = train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# Step 4: Check for missing values
for col in test.columns:
    print(f"'{col}' missing value count: ", test[col].isna().sum())
for col in train.columns:
    print(f"'{col}' missing value count: ", train[col].isna().sum())

# Step 5: Unify title categories
test["title"] = test["title"].replace("Ms", "Miss")
train["title"] = train["title"].replace("Ms", "Miss")

# Step 6: Fill missing ages based on title
title_mapping = {"Mr": 32, "Mrs": 38, "Miss": 21, "Master": 7}
for title, mean_age in title_mapping.items():
    age_mask = (test["Age"].isnull()) & (test["title"] == title)
    test.loc[age_mask, "Age"] = mean_age

title_mapping_train = {"Mr": 32, "Mrs": 35, "Miss": 21, "Master": 4, "Dr": 42}
for title, mean_age in title_mapping_train.items():
    age_mask = (train["Age"].isnull()) & (train["title"] == title)
    train.loc[age_mask, "Age"] = mean_age

# Step 7: Convert Age to integer
test["Age"] = test["Age"].astype("int")
train["Age"] = train["Age"].astype("int")

# Step 8: Fill missing Fare values
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())
train["Fare"] = train["Fare"].fillna(train["Fare"].mean())

# Step 9: Fill missing Embarked values in train data
train["Embarked"].fillna(train["Embarked"].value_counts().idxmax(), inplace=True)

# Step 10: One-hot encoding of categorical variables
pclass_dummies = pd.get_dummies(test["Pclass"], prefix="class", drop_first=True)
sex_dummies = pd.get_dummies(test["Sex"], prefix="sex", drop_first=True)
embarked_dummies = pd.get_dummies(test["Embarked"], prefix="embarked", drop_first=True)
title_dummies = pd.get_dummies(test["title"], prefix="title", drop_first=True)

train_pclass_dummies = pd.get_dummies(train["Pclass"], prefix="class", drop_first=True)
train_sex_dummies = pd.get_dummies(train["Sex"], prefix="sex", drop_first=True)
train_embarked_dummies = pd.get_dummies(train["Embarked"], prefix="embarked", drop_first=True)
train_title_dummies = pd.get_dummies(train["title"], prefix="title", drop_first=True)

test = pd.concat([test, pclass_dummies, sex_dummies, embarked_dummies, title_dummies], axis=1)
train = pd.concat([train, train_pclass_dummies, train_sex_dummies, train_embarked_dummies, train_title_dummies], axis=1)

# Step 11: Drop irrelevant or redundant columns
cols_to_drop = ["Pclass", "Name", "Sex", "SibSp", "Parch", "Ticket", "Cabin", "Embarked", "title"]
test = test.drop(columns=cols_to_drop)
train = train.drop(columns=cols_to_drop)

# Step 12: Convert boolean columns to integers
for col in test.columns:
    if test[col].dtype == bool:
        test[col].replace({True: 1, False: 0}, inplace=True)
for col in train.columns:
    if train[col].dtype == bool:
        train[col].replace({True: 1, False: 0}, inplace=True)

# Step 13: Feature importance using Random Forest
corr_matrix = train.corr(numeric_only=True)
target_corr = corr_matrix["Survived"].sort_values(ascending=False)
print(target_corr)

X = train.drop("Survived", axis=1)
y = train["Survived"]

model = RandomForestRegressor()
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).plot(kind="bar", title="Feature Importance")
plt.tight_layout()
plt.show()

# Step 14: Train logistic regression on selected features
cols_to_use_for_analyse = ["PassengerId", "Fare", "title_Mr", "sex_male", "Age", "class_3"]
X = train[cols_to_use_for_analyse]

logreg = LogisticRegression()
logreg.fit(X, y)
print(f"logistic regression model accuracy: {logreg.score(X, y)}")

# Step 15: Make prediction on test set
test_for_pred = test[X.columns]
predict = logreg.predict(test_for_pred)

# Step 16: Evaluate accuracy
test_survived = df["Survived"]
acc = accuracy_score(predict, test_survived)
print(f"Accuracy score: {acc}")




#%%
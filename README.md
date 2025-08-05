# 🎯 Titanic Survival Prediction with Logistic Regression

This project aims to predict the survival of Titanic passengers using machine learning techniques, particularly logistic regression and random forest models.

## 📌 Project Overview

The dataset is based on the famous Titanic disaster. We preprocess the data by handling missing values, extracting titles from names, engineering features such as passenger class, gender, age, and fare, and then train a classification model to predict survival.

## ⚙️ Key Steps

- Handling missing values (Age, Fare, Embarked)
- Feature engineering: extracting titles (Mr, Mrs, Miss...), converting categorical variables to dummy variables
- Model training using:
  - RandomForestRegressor (for feature importance)
  - LogisticRegression (for prediction)
- Accuracy evaluation with `accuracy_score`

## 🛠️ Technologies Used

- Python
- pandas, NumPy
- seaborn, matplotlib
- scikit-learn (train_test_split, LogisticRegression, RandomForestRegressor, accuracy_score)

## 📊 Model Performance

The logistic regression model is trained on selected features with the highest correlation to survival. The accuracy score is evaluated using the provided test dataset.

## 📁 Dataset

Data used from Kaggle’s [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

- `train.csv`
- `test.csv`
- `gender_submission.csv`

## 🔍 How to Run

1. Clone this repo
2. Make sure required libraries are installed:
   ```bash
   pip install -r requirements.txt

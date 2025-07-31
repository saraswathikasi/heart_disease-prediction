import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

import joblib
import os

# Use your local path to the CSV
data_path = "cleveland_heart_disease_cleaned.csv"  # <-- Make sure this file exists in the same folder
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found!")

df = pd.read_csv(data_path)

y = df["target"]
X = df.drop("target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Define and train classifiers
nb = GaussianNB()
lr = LogisticRegression(max_iter=1000)
svc = SVC(probability=True)

# Create stacking classifier
stack_model = StackingClassifier(
    estimators=[('nb', nb), ('lr', lr), ('svc', svc)],
    final_estimator=LogisticRegression(),
    cv=5
)

stack_model.fit(X_train, y_train)

# Predict and evaluate
scv_predicted = stack_model.predict(X_test)
scv_conf_matrix = confusion_matrix(y_test, scv_predicted)
scv_acc_score = accuracy_score(y_test, scv_predicted)

print("Confusion Matrix:")
print(scv_conf_matrix)
print(f"\nAccuracy of the Stacking Classifier model is: {scv_acc_score * 100:.2f}%\n")
print(classification_report(y_test, scv_predicted))

# Save model
joblib.dump(stack_model, 'stack_model.pkl')

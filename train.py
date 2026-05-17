# Generated from: train.ipynb
# Converted at: 2026-05-16T17:58:36.500Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv("instagram_analytics.csv")
df

df.shape

df.info()

df.head()

df.tail()

df = df.dropna()

print(df.isnull().sum())

target_column = "performance_bucket_label"
target_column

print(df.columns)

label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])


with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

X = df.drop(target_column, axis=1)
y = df[target_column]

categorical_cols = X.select_dtypes(include=['object']).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le


with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib   # ✅ use joblib instead of pickle

# Load dataset
df = pd.read_csv("instagram_analytics.csv")

# Drop unnecessary columns
df = df.drop(columns=[
    'post_id',
    'account_id',
    'post_datetime',
    'post_date',
    'traffic_source',
    'day_of_week'
])

df = df.dropna()
df = df.fillna(0)

# Target column
target_column = "performance_bucket_label"

# Encode target
label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

# Split features/target
X = df.drop(target_column, axis=1)
y = df[target_column]

feature_names = X.columns.tolist()

# Encode categorical columns
categorical_cols = ['account_type', 'media_type', 'content_category']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

bundle = {
    "model": model,
    "label_encoder": label_encoder,
    "encoders": encoders,
    "feature_names": feature_names
}

joblib.dump(bundle, "bundle.pkl", compress=3)

print("All objects saved in bundle.pkl successfully")

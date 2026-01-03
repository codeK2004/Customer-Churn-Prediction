import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/clean_churn.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 🔑 Create separate encoders per column
encoders = {}

for col in X.select_dtypes(include="object"):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le   # save encoder per column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ Model and encoders saved correctly")

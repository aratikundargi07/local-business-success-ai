import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/business_data.csv")

le = LabelEncoder()
df["business_type"] = le.fit_transform(df["business_type"])
df["location_type"] = le.fit_transform(df["location_type"])

X = df.drop("success", axis=1)
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, "model/business_success_model.pkl")
print("Model trained and saved successfully")

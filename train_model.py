# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. DATA LOAD
df = pd.read_csv("heart.csv")

# 2. FEATURES & TARGET
X = df.drop("target", axis=1)
y = df["target"]

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. PIPELINE: SCALER + MODEL (EK HI OBJECT ME)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42
    ))
])

# 5. TRAIN
model.fit(X_train, y_train)

# 6. EVALUATION
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. SAVE SINGLE MODEL FILE
joblib.dump(model, "heart_model.pkl")
print("Saved combined model (scaler + RandomForest) to heart_model.pkl")


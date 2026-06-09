import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
data = pd.read_csv("data.csv", header=None)

# Split features & labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Samples per gesture:")
print(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# RandomForest is much better than KNN for gesture landmarks
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Detailed accuracy per gesture
print("\nAccuracy:", model.score(X_test, y_test))
print("\nPer-gesture report:")
print(classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, "gesture_model.pkl")
print("\nModel saved to gesture_model.pkl")
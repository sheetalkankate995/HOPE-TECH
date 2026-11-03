
import pandas as pd
import os
import sys
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_FOLDER = r"C:\Project by us\SuicidalIdeationProject"
DATASET_FOLDER = os.path.join(PROJECT_FOLDER, "dataset")
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "models")

os.makedirs(MODEL_FOLDER, exist_ok=True)

# Import preprocessing
sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_text

# Load dataset
csv_files = [
    f for f in os.listdir(DATASET_FOLDER)
    if f.endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError("No CSV dataset found!")

data = pd.read_csv(os.path.join(DATASET_FOLDER, csv_files[0]))
data = data.dropna(subset=["text", "class"]).copy()

# Use the same balanced sample
data = data.groupby("class", group_keys=False).sample(
    n=5000, random_state=42
)

data["cleaned_text"] = data["text"].apply(clean_text)

X = data["cleaned_text"]
y = data["class"]

# Split dataset
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

# Load GA vectorizer and selected features
vectorizer = joblib.load(
    os.path.join(MODEL_FOLDER, "ga_vectorizer.pkl")
)

selected_indices = joblib.load(
    os.path.join(MODEL_FOLDER, "selected_features.pkl")
)

# Transform data
X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

X_train_selected = X_train_vectorized[:, selected_indices]
X_test_selected = X_test_vectorized[:, selected_indices]

# Train GA model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_selected, y_train)

# Evaluate model
predictions = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, predictions) * 100

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

labels = model.classes_.tolist()

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print("\n========== GA MODEL RESULTS ==========")
print("Accuracy:", round(accuracy, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(matrix)

# Save GA model
joblib.dump(
    model,
    os.path.join(MODEL_FOLDER, "ga_model.pkl")
)

# Prepare actual evaluation metrics
metrics = {
    "model_name": "Genetic Algorithm Optimized Logistic Regression",
    "accuracy": round(accuracy, 2),
    "precision": round(
        report["weighted avg"]["precision"] * 100, 2
    ),
    "recall": round(
        report["weighted avg"]["recall"] * 100, 2
    ),
    "f1_score": round(
        report["weighted avg"]["f1-score"] * 100, 2
    ),
    "labels": [str(label) for label in labels],
    "confusion_matrix": matrix.tolist(),
    "class_distribution": {
        str(label): int(count)
        for label, count in y_test.value_counts().items()
    },
    "training_records": int(len(y_train)),
    "validation_records": int(len(y_val)),
    "test_records": int(len(y_test)),
    "selected_features": int(len(selected_indices))
}

# Save GA metrics as JSON
metrics_path = os.path.join(
    MODEL_FOLDER,
    "ga_model_metrics.json"
)

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nGA model trained successfully!")
print("Model saved in:", MODEL_FOLDER)
print("Actual GA metrics saved in:", metrics_path)
print("Selected features:", len(selected_indices))

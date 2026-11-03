import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

PROJECT_FOLDER = r"C:\Project by us\SuicidalIdeationProject"
DATASET_FOLDER = os.path.join(PROJECT_FOLDER, "dataset")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "models")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load dataset
csv_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
data = pd.read_csv(os.path.join(DATASET_FOLDER, csv_files[0]))

# Count categories
category_counts = data["class"].value_counts()

# Create pie chart
plt.figure(figsize=(7, 7))

plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Dataset Category Distribution")
plt.axis("equal")

# Save chart
chart_path = os.path.join(RESULTS_FOLDER, "dataset_pie_chart.png")
plt.savefig(chart_path, dpi=300, bbox_inches="tight")

plt.show()

print("Pie chart created successfully!")
print("Saved at:", chart_path)


# Model accuracy comparison
models = ["Logistic Regression", "GA Logistic Regression"]
accuracies = [93.19, 86.95]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"{value}%")

plt.tight_layout()

chart_path = os.path.join(
    RESULTS_FOLDER, "model_accuracy_comparison.png"
)

plt.savefig(chart_path, dpi=300)
plt.show()

print("Accuracy comparison chart saved successfully!")



import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

# Load original model and dataset
model = joblib.load(os.path.join(MODEL_FOLDER, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_FOLDER, "vectorizer.pkl"))

data = data.dropna(subset=["text", "class"]).copy()
data["cleaned_text"] = data["text"].apply(
    lambda text: str(text).lower()
)

# Recreate the original test split
from sklearn.model_selection import train_test_split

X_train_text, X_test_text, y_train, y_test = train_test_split(
    data["cleaned_text"],
    data["class"],
    test_size=0.2,
    random_state=42,
    stratify=data["class"]
)

X_test = vectorizer.transform(X_test_text)
predictions = model.predict(X_test)

# Create heatmap
plt.figure(figsize=(7, 5))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["non-suicide", "suicide"]
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Non-suicide", "Suicide"],
    yticklabels=["Non-suicide", "Suicide"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_FOLDER, "confusion_matrix.png"),
    dpi=300
)

plt.show()

print("Confusion matrix chart saved successfully!")

from sklearn.metrics import roc_curve, roc_auc_score

# Prediction probabilities
probabilities = model.predict_proba(X_test)[:, 1]
y_test_binary = (y_test == "suicide").astype(int)

# Calculate ROC and AUC
fpr, tpr, thresholds = roc_curve(y_test_binary, probabilities)
auc_score = roc_auc_score(y_test_binary, probabilities)

# Create professional ROC graph
plt.figure(figsize=(10, 7))

plt.plot(
    fpr, tpr,
    color="#A855F7",
    linewidth=3,
    label=f"Sentinel AI Model (AUC = {auc_score:.3f})"
)

plt.fill_between(
    fpr, tpr,
    color="#A855F7",
    alpha=0.18
)

plt.plot(
    [0, 1], [0, 1],
    color="gray",
    linestyle="--",
    linewidth=2,
    label="Random Classifier"
)

plt.title(
    "ROC Curve - SENTINEL AI",
    fontsize=18,
    fontweight="bold",
    pad=20
)

plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate", fontsize=13)

plt.xlim(0, 1)
plt.ylim(0, 1.05)

plt.text(
    0.58, 0.18,
    f"AUC SCORE\n{auc_score:.3f}",
    fontsize=16,
    fontweight="bold",
    color="#7E22CE",
    bbox=dict(
        facecolor="#F3E8FF",
        edgecolor="#A855F7",
        boxstyle="round,pad=0.7"
    )
)

plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_FOLDER, "roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Professional ROC Curve created successfully!")
print("AUC Score:", round(auc_score, 3))



from sklearn.metrics import precision_recall_fscore_support

# Calculate performance metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    predictions,
    labels=["non-suicide", "suicide"]
)

# Chart data
categories = ["Non-suicide", "Suicide"]
x = np.arange(len(categories))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1, width, label="F1-Score")

plt.title("Model Performance Metrics - SENTINEL AI", fontsize=16)
plt.xlabel("Categories")
plt.ylabel("Score")
plt.xticks(x, categories)
plt.ylim(0, 1.1)

plt.legend()
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_FOLDER, "performance_metrics.png"),
    dpi=300
)

plt.show()

print("Performance metrics chart created successfully!")

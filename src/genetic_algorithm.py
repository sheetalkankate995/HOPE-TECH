
import pandas as pd
import os
import sys
import random
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Project paths
PROJECT_FOLDER = r"C:\Project by us\SuicidalIdeationProject"
DATASET_FOLDER = os.path.join(PROJECT_FOLDER, "dataset")
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "models")

os.makedirs(MODEL_FOLDER, exist_ok=True)

# Import preprocessing
sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_text

# Set random seeds
random.seed(42)
np.random.seed(42)

# Load dataset
csv_files = [
    f for f in os.listdir(DATASET_FOLDER)
    if f.endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError("No CSV dataset found!")

data = pd.read_csv(os.path.join(DATASET_FOLDER, csv_files[0]))
data = data.dropna(subset=["text", "class"]).copy()

# Balanced sample
data = data.groupby("class", group_keys=False).sample(
    n=5000, random_state=42
)

data["cleaned_text"] = data["text"].apply(clean_text)

X = data["cleaned_text"]
y = data["class"]

# First split: 80% development, 20% final test
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: 60% training, 20% validation, 20% test overall
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

# TF-IDF: fit only on training data
vectorizer = TfidfVectorizer(max_features=300)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)

FEATURE_COUNT = X_train_vectorized.shape[1]

# Genetic Algorithm settings
POPULATION_SIZE = 8
GENERATIONS = 5

def fitness(individual):

    selected = np.where(individual == 1)[0]

    if len(selected) == 0:
        return 0

    model = LogisticRegression(max_iter=500)

    model.fit(
        X_train_vectorized[:, selected],
        y_train
    )

    predictions = model.predict(
        X_val_vectorized[:, selected]
    )

    return accuracy_score(y_val, predictions)

# Create initial population
population = [
    np.random.randint(0, 2, FEATURE_COUNT)
    for _ in range(POPULATION_SIZE)
]

print("\nGenetic Algorithm started!")

# Run Genetic Algorithm
for generation in range(GENERATIONS):

    scores = [fitness(individual) for individual in population]

    best_index = np.argmax(scores)
    best_individual = population[best_index].copy()

    print(
        "Generation:", generation + 1,
        "| Validation Accuracy:", round(scores[best_index], 4),
        "| Selected Features:", int(best_individual.sum())
    )

    new_population = [best_individual.copy()]

    while len(new_population) < POPULATION_SIZE:

        parent1 = random.choice(population)
        parent2 = random.choice(population)

        if FEATURE_COUNT > 1:
            point = random.randint(1, FEATURE_COUNT - 1)
            child = np.concatenate(
                (parent1[:point], parent2[point:])
            )
        else:
            child = parent1.copy()

        # Mutation
        for i in range(FEATURE_COUNT):
            if random.random() < 0.02:
                child[i] = 1 - child[i]

        new_population.append(child)

    population = new_population

# Select best individual using validation data
final_scores = [fitness(individual) for individual in population]

best_individual = population[np.argmax(final_scores)]
selected_indices = np.where(best_individual == 1)[0]

# Save vectorizer and selected features
joblib.dump(
    vectorizer,
    os.path.join(MODEL_FOLDER, "ga_vectorizer.pkl")
)

joblib.dump(
    selected_indices,
    os.path.join(MODEL_FOLDER, "selected_features.pkl")
)

print("\nGENETIC ALGORITHM COMPLETED!")
print("Selected Features:", len(selected_indices))
print("Total Features:", FEATURE_COUNT)
print("Best Validation Accuracy:", round(max(final_scores), 4))
print("Final test data was not used for feature selection.")
print("Selected features saved successfully!")

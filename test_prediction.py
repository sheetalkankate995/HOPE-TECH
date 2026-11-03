
import os
import sys
import joblib

# Project folder
folder = r"C:\Project by us\SuicidalIdeationProject"
models = os.path.join(folder, "models")

# Import preprocessing
sys.path.append(os.path.join(folder, "src"))
from preprocessing import clean_text

# Load models
model = joblib.load(os.path.join(models, "model.pkl"))
vectorizer = joblib.load(os.path.join(models, "vectorizer.pkl"))

ga_model = joblib.load(os.path.join(models, "ga_model.pkl"))
ga_vectorizer = joblib.load(os.path.join(models, "ga_vectorizer.pkl"))
features = joblib.load(os.path.join(models, "selected_features.pkl"))

# Test sentences (not training data)
test_sentences = [
    "I am happy and enjoying my life with my family.",
    "I feel sad and hopeless.",
    "I love my friends and family.",
    "I feel lonely and have no hope."
]

# Test both models
print("\n========== MODEL PROBABILITY RESULTS ==========")

for text in test_sentences:

    cleaned = clean_text(text)

    # Original model
    x = vectorizer.transform([cleaned])
    original_result = model.predict(x)[0]
    original_prob = model.predict_proba(x)[0]

    # GA model
    ga_x = ga_vectorizer.transform([cleaned])
    ga_x = ga_x[:, features]
    ga_result = ga_model.predict(ga_x)[0]
    ga_prob = ga_model.predict_proba(ga_x)[0]

    print("\nSentence:", text)

    print("Original Prediction:", original_result)
    print("Original Probabilities:",
          dict(zip(model.classes_, original_prob.round(3))))

    print("GA Prediction:", ga_result)
    print("GA Probabilities:",
          dict(zip(ga_model.classes_, ga_prob.round(3))))

print("\n========== TESTING COMPLETED ==========")

import joblib
import os

folder = r"C:\Project by us\SuicidalIdeationProject\models"

model = joblib.load(os.path.join(folder, "model.pkl"))
ga_model = joblib.load(os.path.join(folder, "ga_model.pkl"))

print("Original Model Labels:", model.classes_)
print("GA Model Labels:", ga_model.classes_)



import pandas as pd
import os

# Find the project folder
project_folder = r"C:\Project by us\SuicidalIdeationProject"

# Find the CSV file inside dataset folder
dataset_folder = os.path.join(project_folder, "dataset")

csv_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]

if not csv_files:
    print("ERROR: No CSV file found in dataset folder.")
else:
    csv_path = os.path.join(dataset_folder, csv_files[0])

    print("Loading dataset...")
    data = pd.read_csv(csv_path)

    print("\nDATASET LOADED SUCCESSFULLY!")
    print("\nFirst 5 rows:")
    print(data.head())

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nDataset shape (rows, columns):")
    print(data.shape)

    print("\nCategory counts:")
    print(data["class"].value_counts())

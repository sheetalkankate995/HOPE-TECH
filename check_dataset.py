
import os
import pandas as pd
import glob

folder = r"C:\Project by us\SuicidalIdeationProject"
dataset_folder = os.path.join(folder, "dataset")

files = glob.glob(os.path.join(dataset_folder, "*.csv"))

if not files:
    print("No CSV dataset found!")
else:
    df = pd.read_csv(files[0])

    print("\nDATASET COLUMNS:")
    print(df.columns.tolist())

    print("\nCLASS LABEL COUNTS:")
    print(df["class"].value_counts())

    print("\nSAMPLE NON-SUICIDE RECORDS:")
    print(df[df["class"] == "non-suicide"].head(5))

    print("\nSAMPLE SUICIDE RECORDS:")
    print(df[df["class"] == "suicide"].head(5))

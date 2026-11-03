
import pandas as pd
import re
import emoji

def clean_text(text):
    text = str(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Convert emojis into meaningful text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Replace underscores and colons with spaces
    text = text.replace("_", " ")
    text = text.replace(":", " ")

    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


print("Text and emoji preprocessing module is ready!")

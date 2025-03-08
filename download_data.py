import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://data.caf.fr/api/explore/v2.1/catalog/datasets/s_ben_dep/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "s_ben_dep.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
TEST_SIZE = 0.2

def download_data():
    """Download dataset if not already present."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        response = requests.get(DATA_URL, stream=True)
        with open(DATA_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

def split_data():
    """Split dataset into train and test sets."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}. Please download it first.")
    
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE, delimiter=',')
    
    print("Splitting dataset...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    print(f"Train and test datasets saved at {DATA_DIR}")

if __name__ == "__main__":
    download_data()
    split_data()

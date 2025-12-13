# Modules/unstructured_processing.py

import pandas as pd
import numpy as np
import pickle
import os

# ----------------------------
# 1️⃣ Data Loading & Preprocessing
# ----------------------------
def load_raw_data(file_path):
    """Load raw medical data from Excel/CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)

def clean_raw_data(df):
    """Clean and preprocess raw medical data."""
    # Example cleaning steps (adapt as needed from your notebook)
    df = df.drop_duplicates()
    df = df.fillna('Unknown')  # Fill NaNs
    # Additional preprocessing (text cleaning, standardization) goes here
    return df

# ----------------------------
# 2️⃣ Feature Engineering
# ----------------------------
def create_knowledge_base(df, save_path=None):
    """Transform data into a knowledge base with one-hot encoding."""
    # Example: one-hot encode columns 'symptom' and 'disease'
    df_encoded = pd.get_dummies(df, columns=['symptom', 'disease'], dummy_na=True)
    
    if save_path:
        df_encoded.to_csv(save_path, index=False)
    
    return df_encoded

# ----------------------------
# 3️⃣ Artifact Saving
# ----------------------------
def save_artifacts(df, csv_path=None, pkl_path=None):
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV saved to {csv_path}")
    if pkl_path:
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"✅ Pickle saved to {pkl_path}")


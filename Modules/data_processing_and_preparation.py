# data_processing_and_preparation.py

import pandas as pd
import numpy as np
import csv
import pickle
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_raw_data(raw_data_path, cleaned_data_path):
    """
    Loads raw data, cleans it by forward-filling diseases,
    and reshapes it into a 'Disease, Symptom, Count' format.
    
    Args:
        raw_data_path (str): Path to the raw input Excel file.
        cleaned_data_path (str): Path to save the intermediate cleaned CSV file.
    """
    print("--- Part 1: Preprocessing Raw Data ---")
    
    # Load the raw data
    df = pd.read_excel("./data/raw_data.xlsx")
    print(f"Loaded raw data. Shape: {df.shape}")

    # Fill all the NaN values with the value from the row above
    # This associates symptoms with the last specified disease
    cleaned_df = df.ffill()
    print("Forward-filled NaN values.")

    def process_name(name_str):
        """Helper function to clean disease and symptom names."""
        # Handles cases where name_str might be float (NaN)
        if not isinstance(name_str, str):
            return []
        
        # Split by underscore and take the second part if UMLS code is present
        data_name = name_str.replace('^', '_').split('_')
        data_list = [name for i, name in enumerate(data_name) if (i + 1) % 2 == 0]
        return data_list

    disease_symptom_dict = defaultdict(list)
    disease_symptom_count = {}
    
    # Process the cleaned DataFrame to build the dictionaries
    current_disease_list = []
    for _, row in cleaned_df.iterrows():
        if pd.notna(row['Disease']) and row['Disease'].strip():
            disease_str = row['Disease']
            current_disease_list = process_name(disease_str)
            count = row['Count of Disease Occurrence']

        if pd.notna(row['Symptom']) and row['Symptom'].strip():
            symptom_str = row['Symptom']
            symptom_list = process_name(symptom_str)
            for d in current_disease_list:
                disease_symptom_dict[d].extend(symptom_list)
                disease_symptom_count[d] = count

    # Save the reshaped data to a new CSV file
    with open(cleaned_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for disease, symptoms in disease_symptom_dict.items():
            count = disease_symptom_count.get(disease, 0)
            for symptom in symptoms:
                writer.writerow([disease, symptom, count])
    
    print(f"Saved cleaned data to '{cleaned_data_path}'")
    return cleaned_data_path

def create_knowledge_base(cleaned_data_path, knowledge_base_path):
    """
    Creates a one-hot encoded knowledge base from the cleaned data,
    where each row is a disease and columns are symptoms.
    
    Args:
        cleaned_data_path (str): Path to the cleaned data CSV.
        knowledge_base_path (str): Path to save the final knowledge base CSV.
    """
    print("\n--- Part 2: Creating One-Hot Encoded Knowledge Base ---")
    
    # Read the cleaned data
    df = pd.read_csv(cleaned_data_path, header=None, encoding='latin1')
    df.columns = ['Disease', 'Symptom', 'Occurrence_Count']
    df.dropna(inplace=True)
    
    # One-Hot Encode the 'Symptom' column
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['Symptom'])
    
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    print(f"One-hot encoded {len(label_encoder.classes_)} unique symptoms.")
    
    # Create a new DataFrame with the one-hot encoded columns
    symptom_cols = np.asarray(label_encoder.classes_)
    df_ohe = pd.DataFrame(onehot_encoded, columns=symptom_cols)
    
    # Concatenate with the 'Disease' column
    df_concat = pd.concat([df['Disease'].reset_index(drop=True), df_ohe], axis=1)
    
    # Group by disease and sum the one-hot vectors to get symptom counts per disease
    df_knowledge_base = df_concat.groupby('Disease').sum().reset_index()
    
    # Save the final training dataset (knowledge base)
    df_knowledge_base.to_csv(knowledge_base_path, index=False)
    print(f"Saved knowledge base to '{knowledge_base_path}'. Shape: {df_knowledge_base.shape}")
    return df_knowledge_base

def save_app_artifacts(knowledge_base_df, features_path, disease_list_path):
    """
    Saves the final feature vectors (symptoms) and the list of diseases
    for use in the application.
    
    Args:
        knowledge_base_df (pd.DataFrame): The final knowledge base DataFrame.
        features_path (str): Path to save the symptom feature vectors CSV.
        disease_list_path (str): Path to save the pickled disease list.
    """
    print("\n--- Part 3: Saving Final Artifacts for Application ---")
    
    target_column = 'Disease'
    
    # Features are all columns except 'Disease'
    features = knowledge_base_df.drop(target_column, axis=1)
    # Labels are the disease names
    diseases = knowledge_base_df[target_column]

    # Save the knowledge base features (symptom vectors)
    # NOTE: The notebook saved this as an .xlsx file, overwriting the input.
    # Saving as CSV is generally better practice.
    features.to_csv(features_path, index=False)

    # Save the list of disease names in the exact same order
    with open(disease_list_path, 'wb') as f:
        pickle.dump(diseases.tolist(), f)
        
    print(f"Symptom vectors (features) saved to: {features_path}")
    print(f"Disease list (labels) saved to: {disease_list_path}")
    print("\nPipeline complete. Artifacts are ready for the similarity engine.")

if __name__ == '__main__':
    # Define file paths - Assumes a 'data' subfolder exists
    # Please adjust these paths if your folder structure is different
    RAW_DATA_PATH = 'data/raw_data.xlsx'
    CLEANED_DATA_PATH = 'data/cleaned_data.csv'
    KNOWLEDGE_BASE_PATH = 'data/training_dataset_final.csv' 
    
    # Paths for the final application files
    APP_FEATURES_PATH = 'data/symptom_vectors.csv'
    APP_DISEASE_LIST_PATH = 'notebook/disease_list.pkl'
    
    # Run the full data processing pipeline
    preprocess_raw_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
    knowledge_base = create_knowledge_base(CLEANED_DATA_PATH, KNOWLEDGE_BASE_PATH)
    save_app_artifacts(knowledge_base, APP_FEATURES_PATH, APP_DISEASE_LIST_PATH)
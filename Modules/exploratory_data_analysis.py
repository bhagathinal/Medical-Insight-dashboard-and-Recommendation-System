# exploratory_data_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_plots(cleaned_data_path):
    """
    Loads the cleaned data and generates EDA plots.
    
    Args:
        cleaned_data_path (str): Path to the cleaned data CSV file.
    """
    print("--- Running Exploratory Data Analysis (EDA) ---")
    
    # Read the cleaned data
    try:
        df = pd.read_csv(cleaned_data_path, header=None, encoding='latin1')
        df.columns = ['Disease', 'Symptom', 'Occurrence_Count']
    except FileNotFoundError:
        print(f"Error: Could not find '{cleaned_data_path}'.")
        print("Please run 'data_processing_and_preparation.py' first.")
        return

    # --- Plot 1: Top 10 Common Symptoms Across Top 5 Diseases ---
    print("Generating Clustered Bar Chart...")
    
    # Identify top 5 diseases by total occurrence count
    top_diseases = (
        df.groupby('Disease')['Occurrence_Count']
        .first() # Use first since count is the same for all symptoms of a disease
        .sort_values(ascending=False)
        .head(5)
        .index
    )
    
    # Filter the dataframe to include only top 5 diseases
    filtered_df = df[df['Disease'].isin(top_diseases)]
    
    # Identify top 10 most common symptoms across these top diseases
    common_symptoms = (
        filtered_df['Symptom']
        .value_counts()
        .head(10)
        .index
    )
    
    # Filter the data to include only these common symptoms
    final_df = filtered_df[filtered_df['Symptom'].isin(common_symptoms)]
    
    # Plot clustered bar chart
    plt.figure(figsize=(14, 10))
    sns.barplot(
        data=final_df, 
        x='Occurrence_Count', 
        y='Symptom', 
        hue='Disease', 
        dodge=True, 
        palette='Set2'
    )
    plt.title('Occurrence Count of Top Diseases for the 10 Most Common Symptoms')
    plt.xlabel('Disease Occurrence Count')
    plt.ylabel('Symptom')
    plt.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('symptoms_across_diseases.png') # Save the plot
    plt.show()

    # --- Plot 2: Histogram of Disease Occurrence Counts ---
    print("\nGenerating Histogram of Disease Occurrence Counts...")
    
    # Get unique disease counts to avoid summing duplicates
    disease_counts = df[['Disease', 'Occurrence_Count']].drop_duplicates()
    
    plt.figure(figsize=(12, 7))
    sns.histplot(disease_counts['Occurrence_Count'], bins=30, kde=True)
    plt.title('Distribution of Disease Occurrence Counts')
    plt.xlabel('Occurrence Count')
    plt.ylabel('Number of Diseases')
    plt.tight_layout()
    plt.savefig('disease_occurrence_distribution.png') # Save the plot
    plt.show()
    
    print("\nEDA complete. Plots have been displayed and saved as PNG files.")

if __name__ == '__main__':
    # This script depends on the output of the data_processing script
    CLEANED_DATA_PATH = 'data/cleaned_data.csv'
    generate_plots(CLEANED_DATA_PATH)
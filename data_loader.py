import pandas as pd

def load_stss_data(filepath="STSS-131.csv"):
    """
    Load the STSS-131 dataset from a CSV file.
    
    Parameters:
        filepath (str): Path to the STSS-131.csv file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None



def load_semeval_data(filepath="afr_test.csv"):
    """
    Load the SemEval 2024 dataset from a CSV file.

    Parameters:
        filepath (str): Path to the SemEval CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with separate columns for pairs of sentences.
    """
    data = pd.read_csv(filepath)
    
    # Split sentences on the first newline character and assign to two new columns
    sentences = data['Text'].str.split('\n', n=1, expand=True)
    data['Sentence1'] = sentences[0]
    data['Sentence2'] = sentences[1]
    
    return data
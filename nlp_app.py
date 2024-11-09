import streamlit as st
from scipy.stats import pearsonr
from similarity import calculate_sim1, calculate_sim2, calculate_characterbert_similarity
from data_loader import load_stss_data, load_semeval_data
import pandas as pd


# Functions to evaluate correlation
def evaluate_sim1_against_human(data):
    sim1_scores = []
    human_ratings = (data["Human_ratings_similarity"] / 4).tolist()  # Normalize human ratings to 0-1 range
    for idx, row in data.iterrows():
        sentence1 = row["Sentence1"]
        sentence2 = row["Sentence2"]
        sim1_score = calculate_sim1(sentence1, sentence2)
        sim1_scores.append(sim1_score)
    correlation, _ = pearsonr(sim1_scores, human_ratings)
    return correlation

def evaluate_sim2_against_human(data):
    sim2_scores = []
    human_ratings = (data["Human_ratings_similarity"] / 4).tolist()  # Normalize human ratings to 0-1 range
    for idx, row in data.iterrows():
        sentence1 = row["Sentence1"]
        sentence2 = row["Sentence2"]
        sim2_score = calculate_sim2(sentence1, sentence2)
        sim2_scores.append(sim2_score)
    correlation, _ = pearsonr(sim2_scores, human_ratings)
    return correlation

# Function to evaluate CharacterBERT correlation with human ratings
def evaluate_characterbert_against_human(data):
    characterbert_scores = []
    human_ratings = (data["Human_ratings_similarity"] / 4).tolist()  # Normalize human ratings to 0-1 range
    for idx, row in data.iterrows():
        sentence1 = row["Sentence1"]
        sentence2 = row["Sentence2"]
        characterbert_score = calculate_characterbert_similarity(sentence1, sentence2)
        characterbert_scores.append(characterbert_score)
    correlation, _ = pearsonr(characterbert_scores, human_ratings)
    return correlation


# App title
st.title("Semantic Similarity Calculator")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])


# Load dataset options
dataset_option = st.selectbox("Choose Dataset", ("STSS-131", "SemEval 2024 AFR"))

# Load the selected dataset
if dataset_option == "STSS-131":
    data = load_stss_data()
elif dataset_option == "SemEval 2024 AFR":
    data = load_semeval_data("afr_test.csv")

if data is not None:
    st.write(f"Loaded {dataset_option} Dataset:")
    st.write(data.head())


    # Check if the dataset is SemEval, which lacks human ratings
    if dataset_option == "SemEval 2024 AFR":
        st.write("Calculating CharacterBERT Similarity for SemEval 2024 AFR Dataset:")
        
        semeval_scores = []
        for idx, row in data.iterrows():
            sentence1 = row["Sentence1"]
            sentence2 = row["Sentence2"]
            score = calculate_characterbert_similarity(sentence1, sentence2)
            semeval_scores.append({"PairID": row["PairID"], "CharacterBERT_Similarity": score})
        
        # Display results
        st.write("CharacterBERT Similarity Scores for SemEval Dataset:")
        st.write(pd.DataFrame(semeval_scores))

    else:

        # Choose similarity method for Pearson correlation calculation
        method = st.selectbox("Choose Similarity Method for Correlation", ("Sim1", "Sim2", "CharacterBERT"))
        if st.button("Calculate Correlation with Human Ratings"):
            if method == "Sim1":
                correlation = evaluate_sim1_against_human(data)
                st.write("Pearson Correlation between Sim1 and Normalized Human Ratings:", correlation)
            elif method == "Sim2":
                correlation = evaluate_sim2_against_human(data)
                st.write("Pearson Correlation between Sim2 and Normalized Human Ratings:", correlation)
            elif method == "CharacterBERT":
                correlation = evaluate_characterbert_against_human(data)
                st.write("Pearson Correlation between CharacterBERT and Normalized Human Ratings:", correlation)
else:
    st.write("Failed to load dataset.")

# Input fields for sentences and similarity calculation
st.write("Calculate Similarity between Two Sentences:")
method = st.selectbox("Choose Similarity Method", ("Sim1", "Sim2", "CharacterBERT"), key="similarity_method")
sentence1 = st.text_input("Enter Sentence 1:", key="sentence1")
sentence2 = st.text_input("Enter Sentence 2:", key="sentence2")

# Calculate button
if st.button("Calculate Similarity"):
    if method == "Sim1":
        score = calculate_sim1(sentence1, sentence2)
        st.write("Sim1 Score:", score)
    elif method == "Sim2":
        score = calculate_sim2(sentence1, sentence2)
        st.write("Sim2 Score:", score)
    elif method == "CharacterBERT":
        score = calculate_characterbert_similarity(sentence1, sentence2)
        st.write("CharacterBERT Similarity Score:", score)

import sys
import os
sys.path.append(os.path.abspath("character-bert"))


from nltk.corpus import wordnet as wn
from itertools import product
import numpy as np
import spacy
from spacy import load
from transformers import AutoModel
import torch
from transformers import BertTokenizer
from torch.nn.functional import cosine_similarity

from modeling.character_bert import CharacterBertModel  # Import after setting path
from modeling.utils_character_cnn import CharacterIndexer
#from utils.character_cnn import CharacterIndexer


spacy.cli.download("en_core_web_sm")


import nltk
nltk.download('wordnet')
nlp = load("en_core_web_sm")



def custom_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if (norm_vec1 * norm_vec2) != 0 else 0


def calculate_sim1(sentence1, sentence2):
    """
    Calculate the semantic similarity between two sentences using WordNet's Wu-Palmer similarity.

    Parameters:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The average similarity score between nouns in the two sentences.
    """
    # Tokenize sentences and find nouns
    tokens1 = [word for word in sentence1.split() if wn.synsets(word, pos=wn.NOUN)]
    tokens2 = [word for word in sentence2.split() if wn.synsets(word, pos=wn.NOUN)]
    
    # Calculate pairwise Wu-Palmer similarity
    similarities = []
    for word1 in tokens1:
        synsets1 = wn.synsets(word1, pos=wn.NOUN)
        for word2 in tokens2:
            synsets2 = wn.synsets(word2, pos=wn.NOUN)
            max_sim = max((s1.wup_similarity(s2) or 0) for s1, s2 in product(synsets1, synsets2))
            similarities.append(max_sim)
    
    # Return the average similarity
    return np.mean(similarities) if similarities else 0.0




def calculate_sim2(sentence1, sentence2):
    # Process sentences, extract named entities
    doc1, doc2 = nlp(sentence1), nlp(sentence2)
    named_entities1 = [ent.vector for ent in doc1.ents]
    named_entities2 = [ent.vector for ent in doc2.ents]

    # Calculate entity similarity if both sentences contain named entities
    if named_entities1 and named_entities2:
        entity_sims = [
            custom_cosine_similarity(ent1, ent2)
            for ent1 in named_entities1 for ent2 in named_entities2
        ]
        named_entity_similarity = max(entity_sims) if entity_sims else 0
    else:
        named_entity_similarity = 0
    
    # Handling negation and noun conversion
    tokens1 = [convert_to_antonym_if_negated(token) for token in doc1]
    tokens2 = [convert_to_antonym_if_negated(token) for token in doc2]

    # Calculate Wu-Palmer similarity on noun pairs
    similarities = []
    for n1 in tokens1:
        for n2 in tokens2:
            synsets1 = wn.synsets(n1)
            synsets2 = wn.synsets(n2)
            if synsets1 and synsets2:  # Ensure both nouns have synsets
                max_sim = max((s1.wup_similarity(s2) or 0) for s1, s2 in product(synsets1, synsets2))
                similarities.append(max_sim)

    # Compute average similarity if any valid similarities were found
    avg_similarity = np.mean(similarities) if similarities else 0

    # Convex combination of named entity and noun similarity
    combined_similarity = 0.5 * avg_similarity + 0.5 * named_entity_similarity
    return combined_similarity

def convert_to_antonym_if_negated(token):
    """
    Convert an adjective or adverb to its antonym if preceded by negation.
    Parameters:
        token (spacy token): The token to check and potentially modify.
    Returns:
        str: The original token or its antonym if applicable.
    """
    # Check for negation in preceding words
    if token.dep_ == "neg" and token.pos_ in ["ADJ", "ADV"]:
        synsets = wn.synsets(token.text, pos=wn.ADJ if token.pos_ == "ADJ" else wn.ADV)
        for syn in synsets:
            for lemma in syn.lemmas():
                if lemma.antonyms():  # Check if antonym exists
                    return lemma.antonyms()[0].name()  # Return the antonym
    return token.text

def convert_to_noun(token):
    """
    Convert a verb, adjective, or adverb to a noun form.
    Parameters:
        token (spacy token): The token to convert.
    Returns:
        str: The noun form of the token, or the original token if no noun form is available.
    """
    synsets = wn.synsets(token.text, pos=get_wordnet_pos(token))
    for syn in synsets:
        for lemma in syn.lemmas():
            related_forms = lemma.derivationally_related_forms()
            for related in related_forms:
                if related.synset().pos() == wn.NOUN:
                    return related.name()  # Return the first noun form found
    return token.text



def get_wordnet_pos(spacy_token):
    """Helper function to map spaCy pos tags to WordNet pos tags."""
    if spacy_token.pos_ == "VERB":
        return wn.VERB
    elif spacy_token.pos_ == "ADJ":
        return wn.ADJ
    elif spacy_token.pos_ == "ADV":
        return wn.ADV
    return wn.NOUN  # Default to NOUN if no match





# Initialize the BertTokenizer and CharacterIndexer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
indexer = CharacterIndexer()

# Load the pre-trained CharacterBERT model
model_path = './character-bert/pretrained-models/general_character_bert/'
model = CharacterBertModel.from_pretrained(model_path)


def get_characterbert_embeddings(text):
    # Tokenize the text
    tokens = tokenizer.basic_tokenizer.tokenize(text)
    
    # Add [CLS] and [SEP] tokens
    tokens = ['[CLS]', *tokens, '[SEP]']
    
    # Convert token sequence to character indices
    batch = [tokens]  # Create a batch with a single sentence
    batch_ids = indexer.as_padded_tensor(batch)  # Convert tokens to character indices
    
    # Get embeddings from CharacterBERT
    with torch.no_grad():
        embeddings_for_batch, _ = model(batch_ids)
    
    # Extract embeddings for the specific input
    embeddings_for_text = embeddings_for_batch[0]
    
    # Return embeddings (optional: convert to numpy or process further as needed)
    return embeddings_for_text



def calculate_characterbert_similarity(sentence1, sentence2):
    # Get embeddings for each sentence
    embedding1 = get_characterbert_embeddings(sentence1)
    embedding2 = get_characterbert_embeddings(sentence2)
    
    # Average embeddings across tokens to get a single vector per sentence
    embedding1 = embedding1.mean(dim=0)  # Now 1D (768,)
    embedding2 = embedding2.mean(dim=0)  # Now 1D (768,)
    
    # Calculate cosine similarity between the two embeddings
    similarity_score = cosine_similarity(embedding1, embedding2, dim=0).item()
    return similarity_score
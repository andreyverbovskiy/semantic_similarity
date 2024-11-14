# Dockerfile

# Use an official Python image as the base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (WordNet) and SpaCy model needed for similarity calculations
RUN python -m nltk.downloader wordnet
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "nlp_app.py", "--server.port=8501", "--server.enableCORS=false"]

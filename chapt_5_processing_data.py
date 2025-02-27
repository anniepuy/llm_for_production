"""
Title: Chapt 5 - Preprocessing CSV data
Author: Ann Hagan
Date: 2-18-2025
overview: modifications to work with Ollama Mistral locally and some typos from book
requires ollama pull gte-small
"""

import os
import ollama
import requests
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np

# Set up model using Ollama locally
response = ollama.generate(model='mistral')

# Step 1: Download the data
url = 'https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data/mini-llama-articles.csv'
response = requests.get(url)

# Create the data folder if it does not exist
os.makedirs('data', exist_ok=True)

# Save the downloaded data to the folder
file_path = os.path.join('data', 'mini-llama-articles.csv')
with open(file_path, 'wb') as f:
    f.write(response.content)

# Step 2: Chunk the data 1024 chunk size with overlap of 128
def split_into_chunks(text, chunk_size=1024, overlap=128):
    chunks = []
    for i in range(0, len(text), chunk_size-overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Step 3: Use Python CSV to read the data and apply chunking function
chunks = []

# Load the data as CSV
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        if idx == 0:
            continue
        chunks.extend(split_into_chunks(row[1]))

print("number of articles:", idx)
print("number of chunks:", len(chunks))

# Step 4: Save the chunks to a Pandas DataFrame for further processing
df = pd.DataFrame(chunks, columns=['chunk'])
print(df.columns)

# Step 5: Generate Embeddings
# Define embeddings using Ollama's all-minilm model
def get_embedding(text):
    try:
        # Remove newlines and clean text
        text = text.replace('\n', ' ')
        # Call Ollama's embedding endpoint with all-minilm
        response = ollama.embeddings(
            model='all-minilm',
            prompt=text
        )
        # Handle the response correctly
        if isinstance(response, dict) and 'embeddings' in response:
            return response['embeddings']
        return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Generate embeddings with progress bar
print("Generating embeddings...")
embeddings = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    embedding = get_embedding(row['chunk'])
    if embedding is not None:
        embeddings.append(embedding)

# Add embedding column to dataframe
embeddings_values = pd.Series(embeddings)
df.insert(loc=1, column='embedding', value=embeddings_values)
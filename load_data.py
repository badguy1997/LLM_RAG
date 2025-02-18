import json
import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 🔹 Step 1: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 🔹 Step 2: Initialize ChromaDB with Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# 🔹 Step 3: Load Data from JSON File
def load_data_from_json(json_file):
    logging.info(f"Loading data from {json_file}...")
    with open(json_file, 'r') as file:
        data = json.load(file)
        logging.info("Data loaded successfully.")
        return data

# 🔹 Step 4: Add Data to ChromaDB
def add_data_to_chromadb(data):
    texts = [item["Complex_CoT"] for item in data]  # Extracting the Complex_CoT field
    total_docs = len(texts)
    logging.info(f"Adding {total_docs} documents to ChromaDB...")
    
    for i, text in enumerate(texts):
        vector_db.add_texts(texts=[text])  # Add each text individually
        logging.info(f"Document {i + 1}/{total_docs} added successfully.")
    
    logging.info("All documents added successfully!")

# 🔹 Step 5: Main Function
if __name__ == "__main__":
    json_file_path = "data.json"  # Update this path to your JSON file
    if os.path.exists(json_file_path):
        data = load_data_from_json(json_file_path)
        add_data_to_chromadb(data)
    else:
        logging.error(f"JSON file not found at {json_file_path}. Please check the path.") 
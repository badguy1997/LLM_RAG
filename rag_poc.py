import logging
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
import json
import chromadb
import uuid

# 🔹 Step 1: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 🔹 Step 2: Initialize Multiple LLMs
llm_general = OllamaLLM(model="deepseek-r1:1.5b")  # General model
llm_specialized = OllamaLLM(model="llama3.2:latest")  # Specialized model

# 🔹 Step 3: Initialize ChromaDB with Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# 🔹 Step 4: Define RAG Chains for Each Model
rag_chain_general = RetrievalQA.from_chain_type(llm=llm_general, retriever=vector_db.as_retriever(), chain_type="stuff")
rag_chain_specialized = RetrievalQA.from_chain_type(llm=llm_specialized, retriever=vector_db.as_retriever(), chain_type="stuff")

# # Check if database is empty before adding new documents
# if vector_db._collection.count() == 0:
#     print("Adding new documents to ChromaDB...")
#     vector_db.add_texts([
#         "Retrieval-Augmented Generation (RAG) is an AI framework combining retrieval with text generation.",
#         "RAG improves language model responses by fetching relevant documents before generating text.",
#         "ChromaDB is a vector database that stores and retrieves embeddings for RAG implementations."
#     ])
#     print("Documents added successfully!")
# else:
#     print("Database already contains documents, skipping ingestion.")



# 🔹 Step 5: Check Number of Documents
# print("Number of documents in ChromaDB:", vector_db._collection.count())

# 🔹 Step 6: Load Retriever
# retriever = vector_db.as_retriever()

# 🔹 Step 7: Test Retrieval
# query = "What is RAG?"
# retrieved_docs = retriever.invoke(query)

# print("Retrieved Docs for Query:", retrieved_docs)  # Should NOT be empty

# 🔹 Step 8: Define RAG Chain
# rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# 🔹 Step 9: Query RAG Pipeline
def get_rag_response(query_string):
    logging.info(f"Received query: {query_string}")  # Log the received query

    # Get responses from both models
    response_general = rag_chain_general.invoke({"query": query_string})
    response_specialized = rag_chain_specialized.invoke({"query": query_string})

    # Log the responses for debugging
    logging.info(f"General model response: {response_general}")
    logging.info(f"Specialized model response: {response_specialized}")

    # Combine responses (example: choose the one with the highest confidence or simply concatenate)
    combined_response = f"General: {response_general['result']}\nSpecialized: {response_specialized['result']}"
    
    return combined_response  # Return the combined response

# Example usage
if __name__ == "__main__":
    logging.info("Starting RAG system...")
    print(get_rag_response("What is RAG?"))
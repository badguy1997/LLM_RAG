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

# 🔹 Step 2: Initialize Ollama LLM
llm = OllamaLLM(model="deepseek-r1:1.5b")

# 🔹 Step 3: Initialize ChromaDB with Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# 🔹 Step 4: Define RAG Chain
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever(), chain_type="stuff")

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
    response = rag_chain.invoke({"query": query_string})  # Get the response

    # Check if the response is structured as expected
    if isinstance(response, dict) and "result" in response:
        logging.info(f"Response generated: {response['result']}")  # Log the response
        return response["result"]  # Return the complete response
    else:
        logging.error("Unexpected response format: {}".format(response))  # Log an error if the format is unexpected
        return "An error occurred while processing your request."

# Example usage
if __name__ == "__main__":
    logging.info("Starting RAG system...")
    print(get_rag_response("What is RAG?"))
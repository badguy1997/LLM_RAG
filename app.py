import gradio as gr
import logging
from rag_poc import get_rag_response  # Import the get_rag_response function from rag_poc.py

# 🔹 Step 1: Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a function to handle user queries
def query_rag_system(user_query):
    logging.info(f"User query: {user_query}")  # Log the user query
    response = get_rag_response(user_query)  # Get the full response
    logging.info(f"Response sent: {response}")  # Log the response
    return response  # Return the complete response

# Set up the Gradio interface
iface = gr.Interface(
    fn=query_rag_system,  # Function to call
    inputs=gr.Textbox(label="Enter your query:", placeholder="What is RAG?", lines=1),  # Input box for user query
    outputs=gr.Textbox(label="Response:"),  # Output box for the response
    title="RAG Query System",  # Title of the interface
    description="Ask questions about Retrieval-Augmented Generation (RAG) and get responses based on the knowledge base.",
    allow_flagging="never"  # Optional: Disable flagging
)

# Launch the interface
if __name__ == "__main__":
    logging.info("Launching Gradio interface...")
    iface.launch() 
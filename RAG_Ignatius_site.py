import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
import streamlit as st


def create_vector_database(data_dir, embedding_model):
  """Creates a local vector database using .txt files and Faiss.

  Args:
    data_dir: Directory containing .txt files.
    embedding_model: SentenceTransformer model for generating embeddings.

  Returns:
    Faiss index.
  """

  texts = []
  for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
      with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
        texts.append(file.read())

  embeddings = embedding_model.encode(texts)
  dim = embeddings.shape[1]

  index = faiss.IndexFlatL2(dim)
  index.add(embeddings)

  return index, texts

# Example usage
path_to_files = r"C:\Users\jonat\OneDrive\Documents\Ideas\Church Fathers for RAG\Ignatius"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your desired model
data_dir = path_to_files

index, texts = create_vector_database(data_dir, embedding_model)


# Example search
query = "Baptism"
query_embedding = embedding_model.encode([query])
distances, indices = index.search(query_embedding, k=5)


# Print results
for i in indices[0]:
  print(texts[i])
#Initiate LLM

llm = Ollama(model = "llama3")

# Define your query text
query_text = "What is the role of faith in salvation?"

# Encode the query using the same embedding model
query_embedding = embedding_model.encode([query_text])

# Find the k most similar documents (replace k with your desired number)
k = 10
D, I = index.search(query_embedding, k)

# Access the retrieved documents based on the indices (I)
similar_documents = [texts[i] for i in I.tolist()[0]]

# Print the top k similar documents
for document, distance in zip(similar_documents, D.tolist()[0]):
  print(f"Document: {document[:100]}... (Distance: {distance:.4f})")

# Defining the function that accesses the documents

def find_similar_documents(query_text, embedding_model, index, texts, k=10):
    """
    Find the k most similar documents to the query_text using the provided embedding model and index.

    Parameters:
    query_text (str): The text to query.
    embedding_model: The model used to encode the query and documents.
    index: The index used to search for similar documents.
    texts (list): The list of documents to search within.
    k (int): The number of similar documents to retrieve. Default is 10.

    Returns:
    list: A list of tuples containing the similar documents and their distances.
    """
    # Encode the query using the same embedding model
    query_embedding = embedding_model.encode([query_text])

    # Find the k most similar documents
    D, I = index.search(query_embedding, k)

    # Access the retrieved documents based on the indices (I)
    similar_documents = [texts[i] for i in I.tolist()[0]]

    # Combine the documents with their distances
    results = [(document, distance) for document, distance in zip(similar_documents, D.tolist()[0])]

    return results
def query_llm_with_documents(llm, query, documents):
  """Queries an Ollama LLM with a given query and relevant documents.

  Args:
    llm: The Ollama LLM instance.
    query: The user's query.
    documents: A list of retrieved documents.

  Returns:
    The LLM's response.
  """

  # Construct the prompt
  prompt = f"Here are some relevant documents:{(documents)} Based on these letters written by Ignatius, answer the following question. When answering, provide the titles of the epistles that are referenced in your answer. Question: {query}"

  # Generate response using Ollama
  response = llm(prompt)
  return response

def main():
    st.title("Ignatius Search")
    query = st.text_input("Enter your question for Ignatius:")
    if query:
        similar_documents = find_similar_documents(query, embedding_model, index, texts, k=10)
        st.subheader("Retrieved Documents:")
        for document, distance in similar_documents:
            st.write(f"- {document[:100]}... (Distance: {distance:.4f})")
        response = query_llm_with_documents(llm, query, similar_documents)
        st.subheader("Answer:")
        st.write(response)

if __name__ == "__main__":
    main()
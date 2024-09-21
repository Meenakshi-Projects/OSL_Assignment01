
import wikipediaapi
import faiss
import openai
from sentence_transformers import SentenceTransformer
from level1 import cleaned_documents
import chainlit as cl
import re
import numpy as np
import time
from openai.error import RateLimitError

# Set your OpenAI API key
openai.api_key = 'Paste Your OpenAI API Key'

# Initialize the Wikipedia API with a descriptive user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyProjectName (merlin@example.com)'
)

# Load the pre-trained embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocess text 
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Clean up extra spaces
    return text

# Function to fetch Wikipedia articles for a given topic
def fetch_wikipedia_articles(topic):
    try:
        page = wiki_wiki.page(topic)
        if not page.exists():
            print(f"Wikipedia article for '{topic}' not found.")
            return ""  # Return empty string for missing articles
        print(f"Fetched article for '{topic}': {page.summary[:60]}...")  # Print first 60 characters of summary
        return page.summary
    except Exception as e:
        print(f"Error fetching article for '{topic}': {e}")
        return ""

# List of topics to fetch Wikipedia data
topics = ["Artificial Intelligence", "Machine Learning", "Data Science", "Cloud Computing"] 
wikipedia_documents = [fetch_wikipedia_articles(topic) for topic in topics]

# Apply preprocessing to Wikipedia articles
cleaned_wikipedia_documents = [preprocess_text(doc) for doc in wikipedia_documents]

# Combine the documents (20 Newsgroups + Wikipedia)
all_documents = cleaned_documents + cleaned_wikipedia_documents

# Encode all documents into embeddings using the SentenceTransformer model
document_embeddings = embedding_model.encode(all_documents)

# Normalize the embeddings to prepare for cosine similarity search
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Normalize the document embeddings
normalized_document_embeddings = normalize_embeddings(np.array(document_embeddings))

# Function to create a FAISS index for cosine similarity search
def create_faiss_index(embeddings):
    embedding_dimension = embeddings.shape[1]
    # Using the inner product (dot product) for cosine similarity, as vectors are normalized
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)
    return index

# Create FAISS index from the normalized document embeddings
faiss_index = create_faiss_index(normalized_document_embeddings)

# Function to retrieve relevant chunks from FAISS index based on cosine similarity
def get_relevant_faiss_chunks(question, index, top_n=2, threshold=0.5):
    # Encode and normalize the question embedding for cosine similarity
    question_embedding = embedding_model.encode([question])
    normalized_question_embedding = normalize_embeddings(np.array(question_embedding))
    
    # Search the FAISS index for top_n results
    distances, indices = index.search(normalized_question_embedding, top_n)
    
    # Print the cosine similarities for debugging purposes
    print(f"Cosine similarities for '{question}': {distances}")
    
    # Check if chunks were found (higher similarity is better, adjust threshold accordingly)
    if distances[0][0] >= threshold:  # Cosine similarity ranges from -1 to 1
        retrieved_chunks = [all_documents[i] for i in indices[0]]
        print(f"Retrieved chunks: {retrieved_chunks}")
        return retrieved_chunks, True  # True means FAISS retrieved relevant chunks
    else:
        print(f"No relevant chunks found for '{question}'")
        return [], False  # False means no relevant chunks found, fallback to GPT

# Example question for testing
question = "Explain the role of data science in healthcare."

# Adjust the threshold for cosine similarity (higher threshold for stricter matches)
relevant_chunks, from_rag = get_relevant_faiss_chunks(question, faiss_index, threshold=0.5)

# Function to generate an answer using OpenAI's GPT model with RAG
def get_answer_rag(question, context_chunks):
    context = " ".join(context_chunks)
    
    # Token count calculation to adjust the context length
    context_length = len(context.split())
    question_length = len(question.split())
    available_tokens = 2000 - question_length  
    if context_length > available_tokens:
        context = " ".join(context.split()[:available_tokens])

    # Generate the response using OpenAI's GPT model
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful Q&A assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message['content'].strip()


def get_answer_rag_with_retry(question, context_chunks, from_rag, retries=3, delay=5):
    for attempt in range(retries):
        try:
            answer = get_answer_rag(question, context_chunks)
            # Indicate the source of the answer
            if from_rag:
                return f"Answer from RAG (retrieved context): {answer}"
            else:
                return f"Answer from GPT (fallback, no relevant context): {answer}"
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return "Unable to generate an answer due to rate limits. Please try again later."


relevant_chunks, from_rag = get_relevant_faiss_chunks(question, faiss_index)


answer = get_answer_rag_with_retry(question, relevant_chunks, from_rag)

print(f"Answer: {answer}")

#Chainlit Interface
@cl.on_message
async def main(message):
    question = message.content
    
    relevant_chunks, from_rag = get_relevant_faiss_chunks(question, faiss_index)
 
    answer = get_answer_rag_with_retry(question, relevant_chunks, from_rag)
    
    # Send the answer back through Chainlit UI
    await cl.Message(content=f"Answer: {answer}").send()


if __name__ == "__main__":
    cl.run()

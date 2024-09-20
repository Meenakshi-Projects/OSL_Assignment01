import os
import re
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import chainlit as cl


openai.api_key = 'Paste your API KEY'

# Load and clean the data from a text file
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            data = file.read()
        return data.replace('\n', ' ').strip()
    except FileNotFoundError:
        raise Exception(f"File {filepath} not found. Ensure the correct path.")

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'^From:.*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Subject:.*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^X-.*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load  documents
baseball_docs = preprocess_text(load_data(r'Newsgroup_data/rec.sport.baseball.txt'))
hockey_docs = preprocess_text(load_data(r'Newsgroup_data/rec.sport.hockey.txt'))

# Combine documents 
cleaned_documents = [baseball_docs, hockey_docs]

# model for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find relevant document chunks based on a question
def get_relevant_chunks(question, documents, top_n=3, threshold=0.5):
    # Encode the question and documents into embeddings
    question_embedding = embedding_model.encode([question])
    document_embeddings = embedding_model.encode(documents)
    
    # Compute similarity scores using dot product
    scores = np.dot(document_embeddings, question_embedding.T).flatten()
    
   
    if len(scores) == 0:
        return [], None

    
    top_indices = np.argsort(scores)[-top_n:][::-1]
    top_scores = scores[top_indices]

    
    if top_scores[0] > threshold:
        return [documents[i] for i in top_indices], top_scores[0]
    
    return [], None 

# Function to generate an answer using OpenAI directly
def get_answer(question, context_chunks, source="OpenAI"):
    context = " ".join(context_chunks)
    
    # Token count calculation to adjust context length
    context_length = len(context.split())
    question_length = len(question.split())
    available_tokens = 8000 - question_length 
    if context_length > available_tokens:
        context = " ".join(context.split()[:available_tokens])

    # Call OpenAI API directly for answer generation
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. Source: {source}"},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message['content'].strip()

# Example usage
question = "What is the role of a pitcher in baseball?"
relevant_chunks, score = get_relevant_chunks(question, cleaned_documents, top_n=3, threshold=0.5)

# Check if we got relevant chunks from the sports data
if relevant_chunks:
    print(f"Answer is generated from Sports Data (score: {score})")
    answer = get_answer(question, relevant_chunks, source="Sports Data")
else:
    print("No relevant data found, fallback to OpenAI.")
    answer = get_answer(question, [], source="OpenAI")

print(f"Answer: {answer}")

# Chainlit UI integration
@cl.on_message
async def main(message):
    question = message.content
    relevant_chunks, score = get_relevant_chunks(question, cleaned_documents, top_n=3, threshold=0.5)

    if relevant_chunks:
        answer = get_answer(question, relevant_chunks, source="Sports Data")
    else:
        answer = get_answer(question, [], source="OpenAI")
    
    await cl.Message(content=f"Answer: {answer}").send()

# Entry point for runChainlit UI
if __name__ == "__main__":
    cl.run()

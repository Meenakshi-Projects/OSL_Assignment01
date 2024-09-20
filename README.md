# OSL_Assignment01

# QA System with RAG Approach

## Overview

This project implements a question-answering (QA) system using Retrieval-Augmented Generation (RAG). It integrates document retrieval, embedding-based similarity, and OpenAI’s GPT models to provide intelligent, context-aware answers. The system is built using Python and Chainlit for the UI, along with OpenAI for answer generation.

## Project Structure
Newsgroup_data/
- rec.sport.baseball.txt
- rec.sport.hockey.txt
level1.py # Main code
documentation-level1.docx # detailed documentation
README.md # Documentation


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/rag-qa-system.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Add your OpenAI API key in `level1.py`:

    ```python
    openai.api_key = 'your-api-key'
    ```

4. Run the application:

    ```bash
    python level1.py
    ```

## Features

- **Data Preprocessing**: Load and clean text data from the 20 Newsgroups dataset.
- **Embedding Generation**: Uses SentenceTransformer to generate embeddings.
- **Document Retrieval**: Retrieves relevant documents based on semantic similarity.
- **Answer Generation**: Generates answers using OpenAI's GPT-3.5-turbo model.
- **UI Integration**: Provides an interactive UI via Chainlit for asking questions and receiving answers.

## Future Work

- **Vector Store**: Integrate FAISS or Pinecone for scalable vector search.
- **Data Sources**: Add more datasets (e.g., Wikipedia) and enable topic-based querying.
- **Containerization**: Dockerize the application for easy deployment.

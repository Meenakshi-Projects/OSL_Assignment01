# OSL_Assignment01

# QA System with RAG Approach using Newsgroup data and Wikipedia Article 

## Overview

This project implements a question-answering (QA) system retrieves relevant context from Wikipedia and a local dataset, using Retrieval-Augmented Generation (RAG). It integrates document retrieval, embedding-based similarity, and OpenAI’s GPT models to provide intelligent, context-aware answers. The system is built using Python and Chainlit for the UI, along with OpenAI for answer generation.

## Project Structure
Newsgroup_data/
- rec.sport.baseball.txt
- rec.sport.hockey.txt
level1.py and level_2.py # Main code
requirements.txt # libraries and dependencies project needs
documentation-level1.docx/ documentation-level1 # detailed documentation
README.md # Documentation


## Installation

1. Clone the repository:

    git clone https://github.com/Meenakshi-Projects/OSL_Assignment01.git


2. Install the required dependencies:

    pip install

3. Add your OpenAI API key in `level1.py`:

    openai.api_key = 'your-api-key'
   
5. Install all the dependencies using 
requirements.txt

4. Run the application:

    chainlit run [level1.py, level_2.py]

## Features

- **Data Preprocessing**: Load and clean text data from the 20 Newsgroups dataset.
- **Embedding Generation**: Uses SentenceTransformer to generate embeddings.
- **Document Retrieval**: Retrieves relevant documents based on semantic similarity.
- **Answer Generation**: Generates answers using OpenAI's GPT-3.5-turbo and GPT-4 model.
- **UI Integration**: Provides an interactive UI via Chainlit for asking questions and receiving answers.
- **Vector Store**: Integrate FAISS for scalable vector search, used cosine similarity to find best match.
- **Data Sources**: Add more datasets (Wikipedia) and enabled topic-based querying.
- **Labaled data sourc** : Whether data coming from RAG or GPT

## Future Work
- **Containerization**: Dockerize the application for easy deployment.

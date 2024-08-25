# Finance-Related Chatbot with RAG By Vivek Nair

This project is a Flask-based chatbot that answers finance-related queries using a Retrieval-Augmented Generation (RAG) approach. It utilizes sentence embeddings, document splitting, and a vector store for document retrieval. The chatbot provides context-aware responses based on uploaded documents.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [API Endpoints](#api-endpoints)

## Installation

### Prerequisites
- Python 3.x
- `pip` package installer
- CUDA-compatible GPU (for running SentenceTransformer on GPU)
- [Chromadb](https://pypi.org/project/chromadb/)
- [SentenceTransformers](https://pypi.org/project/sentence-transformers/)
- [Langchain](https://pypi.org/project/langchain/)
- [Ollama](https://pypi.org/project/ollama/)
- pip install pip install Flask flask-cors chromadb sentence-transformers langchain langchain_community  ollama
- Download ollama.exe from https://ollama.com/download once downloaded install it which will open up a command prompt.
- Run the following command to download llama2 (ollama run llama2) in command prompt

### Steps
1. Ensure the `uploads` directory exists for storing uploaded documents:
   The document you want for the RAG should be inside the uploads folder in the same directory

## Usage

To start the Flask application, run:
python app.py

This will start the application in development mode. Navigate to http://127.0.0.1:5000/ in your web browser to access the chatbot UI.

Please be patient while waiting for response as it might take upto 2mins to generate a response as its running locally.

## Features

Document Upload: Upload PDF and text documents for the chatbot to use as context.

Finance-Related Queries: The chatbot answers only finance-related questions.

Session Management: Keeps track of user sessions and conversation history.

RAG (Retrieval-Augmented Generation): Combines retrieved documents with generated responses.

## Api Endpoints

GET /
Serves the chatbot UI.

POST /ask
Processes a user question and returns the chatbot response.

Request Body: JSON object with a question field.
Response: JSON object with a response field containing the chatbot's answer.

POST /clear_session
Clears the user session and conversation history.
Request Body: Empty.
Response: JSON object with a message field confirming session clearance.
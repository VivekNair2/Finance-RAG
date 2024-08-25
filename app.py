from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import chromadb
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import uuid
from dotenv import load_dotenv
import ollama

app = Flask(__name__)
CORS(app)
load_dotenv()
app.secret_key = os.urandom(24)  


collection = None 
user_sessions = {}  


client = chromadb.Client()


def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            continue 
        documents.extend(loader.load())
    return documents

# Initialization tasks
def initialize_app():
    global collection
   
    print("Initialization tasks before first request.")

    try:
       
        uploads_directory = 'uploads'
        docs = load_documents_from_directory(uploads_directory)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        embeddings = [model.encode(doc.page_content, convert_to_tensor=True).cpu().numpy().tolist() for doc in splits]

       
        collection_name = "Uploaded_Documents"  
        collection = client.create_collection(name=collection_name)

        
        for doc, embedding in zip(splits, embeddings):
            doc_id = str(uuid.uuid4())
            collection.add(
                ids=[doc_id],
                documents=[doc.page_content],
                metadatas=[{"source": doc.metadata["source"]}],
                embeddings=[embedding]
            )

        print(f"Documents from '{uploads_directory}' loaded and processed successfully.")

    except Exception as e:
        
        print(f"Error during initialization: {str(e)}")


initialize_app()


@app.route('/')
def home():
    return render_template('index.html')


def load_finance_keywords(file_path):
    with open(file_path, 'r') as file:
        keywords = file.read().splitlines()
    return keywords

def is_finance_related(question, finance_keywords):
    question = question.lower()
    for keyword in finance_keywords:
        if keyword.lower() in question:
            return True
    return "Uncertain"


@app.route('/ask', methods=['POST'])
def rag_chain():
    global collection
    data = request.get_json()
    question = data.get('question', '')
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    if session_id not in user_sessions:
        user_sessions[session_id] = []
    try:
        response = process_question(question, session_id)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})

def process_question(question, session_id):
    global collection
    if collection:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        query_embedding = model.encode(question, convert_to_tensor=True).cpu().numpy().tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        if results and results.get('documents'):
            retrieved_docs = [{"page_content": doc, "metadata": {"source": meta["source"]}} 
                              for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
            formatted_context = "\n\n".join(doc["page_content"] for doc in retrieved_docs)
            conversation_history = user_sessions[session_id]
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant that answers only  finance-related questions based on the given context and also  looking at conversation history. If the question is not clearly finance-related and also out of the context reply politely  that you are Barbarik AI chatbot made for only financial related queries. If you cannot provide a finance-related answer, politely explain that you specialize in finance and may not be able to provide accurate information on other topics.'}
            ]
            messages.extend(conversation_history)
            messages.append({'role': 'user', 'content': f"Question: {question}\nIs Finance-Related: {'Yes' if is_finance_related(question, load_finance_keywords('finance_keywords.txt')) else 'Possibly Not'}\n\nContext: {formatted_context}"})
            response = ollama.chat(model='llama2', messages=messages)
            user_sessions[session_id].append({'role': 'user', 'content': question})
            user_sessions[session_id].append({'role': 'assistant', 'content': response['message']['content']})
            user_sessions[session_id] = user_sessions[session_id][-10:]
            return response['message']['content']
        else:
            return "I don't have enough information to answer this question accurately."
    else:
        return "Please upload a document first."

@app.route('/clear_session', methods=['POST'])
def clear_session():
    session_id = session.get('session_id')
    if session_id in user_sessions:
        del user_sessions[session_id]
    session.clear()
    return jsonify({"message": "Session cleared successfully"})

if __name__ == '__main__':
    app.run(debug=True)
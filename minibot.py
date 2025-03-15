import datetime
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

import os
import unstructured
 

# Load and process documents from the "data" directory
def load_documents():
    loader = DirectoryLoader("data", glob="*.txt")  # Load text files from data directory
    documents = loader.load()
    for doc in documents:
        filename = os.path.basename(doc.metadata["source"])
        if filename.startswith("menu_") and filename.endswith(".txt"):
            try:
                date_str = filename[5:-4]  # Extract date part
                doc.metadata["date"] = datetime.datetime.strptime(date_str, "%Y_%m_%d").date()
            except ValueError:
                doc.metadata["date"] = "Unknown"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Save documents to a local FAISS database
def create_vector_db():
    documents = load_documents()
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")  # Use Ollama for embeddings
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("file_db")  # Save locally

def load_vector_db():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    if os.path.exists("file_db"):
        return FAISS.load_local("file_db", embeddings,allow_dangerous_deserialization=True)
    else:
        create_vector_db()
        return FAISS.load_local("file_db", embeddings,allow_dangerous_deserialization=True)


# Initialize Ollama LLM
llm = ChatOllama(model="mistral")  # Change "mistral" to your preferred model


# Initialize chat history memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



# Load vector database for RAG retrieval
vector_db = load_vector_db()
retriever = vector_db.as_retriever()


chain = chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,  # Use FAISS as the retriever
    memory=memory
)



def chat_with_bot(user_input, history):
    """Process user input and return chatbot response."""
    response = chain.invoke({"question": user_input}, config={"configurable": {"session_id": 1}})
    return response["answer"]

# Create Gradio chat interface
demo = gr.ChatInterface(fn=chat_with_bot, title="Ollama Chatbot")

demo.launch()
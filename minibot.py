from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import gradio as gr


# Initialize Ollama LLM
llm = ChatOllama(model="mistral")  # Change "mistral" to your preferred model

# Initialize chat history memory
store = {}  # memory is maintained outside the chain

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chain = RunnableWithMessageHistory(llm, get_session_history)

def chat_with_bot(user_input, history):
    """Process user input and return chatbot response."""
    response = chain.invoke({"question": user_input}, config={"configurable": {"session_id": 1}})
    return response.content

# Create Gradio chat interface
demo = gr.ChatInterface(fn=chat_with_bot, title="Ollama Chatbot")

demo.launch()
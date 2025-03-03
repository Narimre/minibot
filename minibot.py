import ollama
import gradio as gr
def chat(user_input,history):
    response = ollama.chat(model="phi3", messages=[{"role": "user", "content": user_input}])
    return response["message"]["content"]
demo = gr.ChatInterface(fn=chat, type="messages", title="Ollama Chatbot")
demo.launch()

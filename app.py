import gradio as gr
from src.chat import Chat
from dotenv import load_dotenv
from src.config import Config
from src.vector_store import VectorStore
import os

load_dotenv()

def main():
    # initialize components
    llm = Config.get_llm()
    vector_store = VectorStore()
    chat = Chat(
        name=os.getenv("NAME"), 
        retriever=vector_store.get_retriever(), 
        llm=llm
    )

    # create gradio interface
    gr.ChatInterface(chat, type="messages").launch()
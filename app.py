import gradio as gr
from src.chat import Chat
from dotenv import load_dotenv
from src.config.config import Config
from src.vector_store.vector_store import VectorStore
import os
from src.document_ingestion.document_processor import DocumentProcessor

load_dotenv()

# Pipeline trace (data flow):
# 1. Config.DEFAULT_URLS + DocumentProcessor.process_url(urls)
#    -> _load_documents: each source = URL (WebBaseLoader) | dir (PyPDFDirectoryLoader) | .pdf (PyPDFLoader) | .txt (TextLoader)
#    -> _split_documents: RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)
#    -> list[Document] (chunks)
# 2. VectorStore.create_retriever(documents) -> FAISS.from_documents + as_retriever()
# 3. Chat(retriever, llm) -> RagNodes(retriever, llm); chat.chat(message, history) -> nodes.generate_answer
# 4. generate_answer: ReAct agent with retriever + wikipedia tools; agent.invoke(messages) -> answer
# 5. gr.ChatInterface(chat.chat) calls chat.chat(message, history) on each user message

def main():
    # initialize components
    llm = Config.get_llm()
    documents = DocumentProcessor().process_url(Config.DEFAULT_URLS)
    vector_store = VectorStore()
    if vector_store.retriever is None:
        vector_store.create_retriever(documents)
    name = os.getenv("NAME", "Assistant")
    chat = Chat(
        name=name,
        retriever=vector_store.get_retriever(),
        llm=llm,
    )

    # create gradio interface
    gr.ChatInterface(chat.chat, textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7)).launch()

if __name__ == "__main__":
    main()
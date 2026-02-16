import os
from openai import OpenAI
from dotenv import load_dotenv

from src.nodes.reactnode import RagNodes
from src.state.rag_state import RagState

load_dotenv()

class Chat:
    def __init__(self, name: str, retriever, llm):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.nodes = RagNodes(retriever=retriever, llm=llm, name=name)

    def chat(self, message, history):
        return self.nodes.generate_answer(RagState(question=message, retrieved_docs=[], answer="")).answer
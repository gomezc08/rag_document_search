import os
from openai import OpenAI
from dotenv import load_dotenv

from src.nodes.reactnode import RagNodes
from langchain_core.messages import HumanMessage

load_dotenv()

SYSTEM_PROMPT = """
You are acting as {name}. You are answering questions on {name}'s website, 
particularly questions related to {name}'s career, background, skills and experience. 
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. 
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. 
Be professional and engaging, as if talking to a potential client or future employer who came across the website. 
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. 
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. 
"""

class Chat:
    def __init__(self, name: str, retriever, llm):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.nodes = RagNodes(retriever, llm)
        self.graph = self.nodes.build_graph()
        self.name = name

    def chat(self, message, history):
        return self.graph.invoke({"messages": history + [HumanMessage(content=message)]})
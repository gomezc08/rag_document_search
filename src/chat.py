import os
from openai import OpenAI
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.nodes.reactnode import RagNodes
from src.state.rag_state import RagState

load_dotenv()

def _gradio_history_to_messages(history: list) -> list[BaseMessage]:
    """Convert Gradio chat history (list of {role, content}) to LangChain messages."""
    if not history:
        return []
    messages = []
    for turn in history:
        role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
        content = turn.get("content") if isinstance(turn, dict) else getattr(turn, "content", "")
        # Normalize content: Gradio can send str or dict (e.g. {"text": "..."}) or list of parts
        if isinstance(content, dict) and "text" in content:
            content = content["text"]
        elif isinstance(content, list):
            parts = [p.get("text", p) if isinstance(p, dict) else str(p) for p in content]
            content = " ".join(parts) if parts else ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        if role == "user":
            messages.append(HumanMessage(content=content or ""))
        elif role == "assistant":
            messages.append(AIMessage(content=content or ""))
    return messages


class Chat:
    def __init__(self, name: str, retriever, llm):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.nodes = RagNodes(retriever=retriever, llm=llm, name=name)

    def chat(self, message: str, history: list):
        prior_messages = _gradio_history_to_messages(history)
        state = RagState(question=message, retrieved_docs=[], answer="")
        return self.nodes.generate_answer(state, prior_messages=prior_messages).answer
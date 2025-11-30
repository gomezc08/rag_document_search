"""RAG state definition for LangGraph"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RagState(BaseModel):
    """State object for RAG worflow"""
    
    question: str
    retrieved_docs: List[Document] = Field(default_factory=list)
    answer: str = ""
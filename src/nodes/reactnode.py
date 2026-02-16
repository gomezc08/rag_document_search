"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
from src.state.rag_state import RagState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

from dotenv import load_dotenv
import os

SYSTEM_PROMPT = """
You are {name}. You are answering questions on {name}'s website about {name}'s career, background, skills and experience.
Your responsibility is to represent {name} faithfully. Use ONLY the "Relevant context from your documents" below (and your retriever tool if you need more detail) to answer. Do not make up information.
Be professional and engaging. If you don't know the answer, say so and offer to connect via email.
"""

load_dotenv()

class RagNodes():
    """Contains node functions for RAG workflow"""
    def __init__(self, retriever, llm, name: Optional[str] = None):
        """
        Initializes RAG nodes

        Args:
            retriever: vector store instance
            llm: llm instance
            name: display name for the person (from .env NAME); passed explicitly so it is correct at runtime
        """
        self.retriever = retriever
        self.llm = llm
        self._agent = None
        self.name = name or os.getenv("NAME") or "Assistant"
        
    def retrieve_docs(self, state:RagState) -> RagState:
        """
        Retreived docs
        
        Args:
            state: Rag state instance
            
        Returns:
            Update RAG state with retrieved docs.
        """
        docs = self.retriever.invoke(state.question)
        return RagState(
            question = state.question,
            retrieved_docs=docs
        ) 
    
    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""
        
        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)
        
        retriever_tool = Tool (
            name = "retriever",
            description = "Fetch passages from indexed vectorstores",
            func = retriever_tool_fn
        )
        
        wiki = WikipediaAPIWrapper(
            api_wrapper = WikipediaAPIWrapper(top_k_results=3, lang = "en")
        )
        wiki_tool = Tool (
            name = "wikipedia",
            description = "Search Wikipedia for general knowledge.",
            func = wiki.run
        )
        
        return [retriever_tool, wiki_tool]
        
    
    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        self._agent = create_agent(self.llm, tools=tools, system_prompt=SYSTEM_PROMPT.format(name=self.name))
    
    def _format_docs_as_context(self, docs: List[Document], max_docs: int = 8) -> str:
        """Format retrieved documents as a single context string for the prompt."""
        if not docs:
            return "(No relevant documents found.)"
        merged = []
        for i, d in enumerate(docs[:max_docs], start=1):
            meta = d.metadata if hasattr(d, "metadata") else {}
            title = meta.get("title") or meta.get("source") or f"doc_{i}"
            merged.append(f"[{i}] {title}\n{d.page_content}")
        return "\n\n".join(merged)

    def generate_answer(self, state: RagState) -> RagState:
        """
        Retrieve relevant docs from the vector store, then generate answer using the agent
        with that context in the prompt so responses are grounded in your documents.
        """
        if self._agent is None:
            self._build_agent()

        # RAG: retrieve docs for this question and inject into the user message
        docs = self.retriever.invoke(state.question)
        context = self._format_docs_as_context(docs)
        user_content = (
            f"Relevant context from your documents:\n{context}\n\n"
            f"User question: {state.question}"
        )

        result = self._agent.invoke({"messages": [HumanMessage(content=user_content)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RagState(
            question=state.question,
            retrieved_docs=docs,
            answer=answer or "Could not generate answer."
        )
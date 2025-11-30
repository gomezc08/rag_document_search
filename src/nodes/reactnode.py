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

class RagNodes():
    """Contains node functions for RAG workflow"""
    def __init__(self, retriever, llm):
        """
        Initializes RAG nodes
        
        Args:
            retriever: vector store instance
            llm: llm instance
        """
        self.retriever = retriever
        self.llm = llm
        self._agent = None
        
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
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )

        self._agent = create_agent(model=self.llm, tools=tools, system_prompt=system_prompt)
    
    def generate_answer(self, state: RagState) -> RagState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            self._build_agent()
        
        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        
        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)
        
        return RagState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )
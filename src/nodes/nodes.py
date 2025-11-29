"""Functionality module for each node"""

import os
from typing import List
from src.vector_store.vector_store import VectorStore
from src.state.rag_state import RagState

class RagNodes:
    """Contains node function"""
    
    def __init__(self, retriever, llm):
        """
        Initializes RAG nodes
        
        Args:
            retriever: vector store instance
            llm: llm instance
        """
        self.retriever = retriever
        self.llm = llm
    
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
    
    def generate_answer(self, state:RagState) -> RagState:
        """
        Generate answer from retrieved docs
        
        Args:
            state: Rag state instance
        
        Returns:
            Update RAG state with answer
        """
        # combine retrieved docs into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        # create prompt
        prompt = f"""Answer the following question based on the context.
        Context: {context}
        
        Question: {state.question}
        """
        
        # generate response.
        response = self.llm.invoke(prompt)
        return RagState(
            question = state.question,
            retrieved_docs = state.retrieved_docs,
            answer = response.content
        )
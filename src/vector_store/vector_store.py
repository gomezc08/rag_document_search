"""Vector store module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """Manages vector store"""
    
    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.vector_store = None
        self.retriever = None
    
    def create_retriever(self, documents:List[Document]):
        """
        Creates vector store from documents
        
        Args:
            documents: List of documents to embed
        """
        self.vector_store = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vector_store.as_retriever()
    
    def get_retriever(self):
        """
        Get retriever instance
        
        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store is not initalized.")
        return self.retriever
    
    def retrieve(self, query:str, k: int = 4) -> List[Document]:
        """
        Grabs k relative docs for a specific query
        
        Args:
            query: user query
            k: number of docs to return
            
        Returns:
            List of retrieved docs
        """
        if self.retriever is None:
            raise ValueError("Vector store is not initalized.")
        return self.retriever.invoke(query)
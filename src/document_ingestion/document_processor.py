"""Document processing module for loading and splitting documents"""

from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size:int = 500, chunk_overlap:int = 50):
        """
        Initalize document processor
        Args:
            chunk_size: size of each chunk for our retriever
            chunk_overlap: how many token overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )
        
    def _load_from_pdf_directory(self, directory:Union[str, Path]) -> list[Document]:
        """Load pdfs from directory"""
        loader = PyPDFDirectoryLoader(directory)
        return loader.load()
    
    def _load_from_pdf(self, file_path:Union[str, Path]) -> list[Document]:
        """Load pdfs from file"""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def _load_from_url(self, url:str) -> List[Document]:
        """Load documents from all urls"""
        loader = WebBaseLoader(url)
        return loader.load()
    
    def _load_from_text(self, file_path:Union[str, Path]) -> list[Document]:
        """Load pdfs from either directory or path"""
        loader = TextLoader(str(file_path, encoding="utf-8"))
        return loader.load()
    
    def _load_documents(self, sources:List[str]) -> List[Document]:
        """
        Load documents from URLs, PDFs, and/or text files
        
        Args:
            sources: List of URLs, PDF folder paths, or text files
            
        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            
            # Case 1: URL
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self._load_from_url(src))
            
            # Case 2: PDF directory
            path = Path("data")
            if path.is_dir():
                docs.extend(self._load_from_pdf_dir(path))
                
            # Case 3: Text file 
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            
            # Case 4: Unavailable source type
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, .txt file, or PDF directory."
                )

        return docs
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: docs we will be splitting
            
        Return:
            List of split docs
        """
        return self.splitter.split_documents(documents)
    
    def process_url(self, urls:List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
        
        Returns:
            List of processed document chunks
        """
        docs = self._load_documents(urls)
        split_docs = self._split_documents(docs)
        return split_docs
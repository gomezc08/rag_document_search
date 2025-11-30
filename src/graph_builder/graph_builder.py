"""Module for creating our graph"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RagState
from src.nodes.reactnode import RagNodes

class GraphBuilder():
    """Builds and manages LangGrapgh"""
    
    def __init__(self, retriever, llm):
        """
        Initalizes graph builder 
        
        Args:
            retriever: doc retriever intstance
            llm: generator 
        """
        self.nodes = RagNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Builds graph
        
        Returns:
            Compiled graph
        """
        # create state graph.
        builder = StateGraph(RagState)
        
        # nodes.
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # set entry point.
        builder.set_entry_point("retriever")
        
        # edges.
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        
        # compile
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question:str) -> dict:
        """
        Run agentic workflow
        
        Args:
            question: user query
        
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        # define inital state.
        inital_state = RagState(question = question)
        
        # invoke graph.
        return self.graph.invoke(inital_state)
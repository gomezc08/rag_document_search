"""Module for creating our graph"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RagState
from src.nodes.reactnode import RagNodes

class GraphBuilder():
    """Builds and manages LangGrapgh"""
    
    def __init__(self, retriever, llm, checkpointer=None):
        """
        Initalizes graph builder 
        
        Args:
            retriever: doc retriever intstance
            llm: generator
            checkpointer: optional checkpointer for graph and agent memory
        """
        self.nodes = RagNodes(retriever, llm, checkpointer=checkpointer)
        self.graph = None
        self.checkpointer = checkpointer
    
    def build(self, checkpointer):
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
        self.graph = builder.compile(checkpointer=checkpointer)
        return self.graph
    
    def run(self, question:str, checkpointer, thread_id: str = "default") -> dict:
        """
        Run agentic workflow
        
        Args:
            question: user query
            checkpointer: checkpointer instance
            thread_id: thread identifier
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build(checkpointer)
        
        # prepare checkpointer.
        config = {"configurable": {"thread_id": thread_id}}

        #  Get previous state if it exists (for conversation continuity)
        previous_state = self.graph.get_state(config)
        if previous_state.values:
            # Preserve previous state, update question and thread_id
            initial_state = RagState(
                question=question,
                retrieved_docs=previous_state.values.get("retrieved_docs", []),
                answer=previous_state.values.get("answer", ""),
                thread_id=thread_id
            )
        else:
            # First message in thread
            initial_state = RagState(question=question, thread_id=thread_id)
        
        # invoke graph - checkpointer will save state automatically
        result = self.graph.invoke(initial_state, config=config)
        
        # Return as dict
        return result if isinstance(result, dict) else result.model_dump()

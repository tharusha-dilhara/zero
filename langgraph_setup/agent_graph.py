from typing import Dict, List, Tuple, Any, Optional
from langgraph.graph import StateGraph, END
import json
from utils.llm_manager import GroqLLMManager
from agents.sales_agent import SalesAgent
from agents.help_agent import HelpAgent
from agents.manage_agent import ManageAgent
from agents.marketing_agent import MarketingAgent
from utils.memory import QdrantMemory

class AgentGraph:
    def __init__(self, llm_manager: GroqLLMManager, memory: QdrantMemory):
        """Initialize agent communication graph."""
        self.llm_manager = llm_manager
        self.memory = memory
        
        # Initialize agents
        self.sales_agent = SalesAgent(llm_manager=llm_manager, memory=memory)
        self.help_agent = HelpAgent(llm_manager=llm_manager, memory=memory)
        self.manage_agent = ManageAgent(llm_manager=llm_manager, memory=memory)
        self.marketing_agent = MarketingAgent(llm_manager=llm_manager, memory=memory)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the agent communication graph."""
        # Define the state
        class AgentState:
            query: str
            agent_responses: Dict[str, str]
            current_agent: str
            conversation_history: List[Dict[str, str]]
            final_response: Optional[str] = None
        
        # Create the state graph
        graph = StateGraph(AgentState)
        
        # Define nodes
        
        # Router node to determine which agent should respond
        def router(state):
            """Route the query to appropriate agent."""
            query = state["query"]
            
            # Use the LLM to determine which agent should handle the query
            router_prompt = f"""Based on the following query, determine which agent should handle it:
            Query: {query}
            
            Available agents:
            - sales: For queries about courses, pricing, and enrollment
            - help: For technical issues, login problems, and general support
            - manage: For operational inquiries and system status
            - marketing: For information about promotions and marketing materials
            
            Reply with just the agent name (sales, help, manage, or marketing):"""
            
            agent = self.llm_manager.generate(router_prompt).strip().lower()
            
            # Fallback to help agent if the determination is unclear
            if agent not in ["sales", "help", "manage", "marketing"]:
                agent = "help"
                
            return {"current_agent": agent}
        
        # Agent nodes
        def sales_node(state):
            """Sales agent node."""
            query = state["query"]
            response = self.sales_agent.process_message(query)
            return {"agent_responses": {**state.get("agent_responses", {}), "sales": response}}
        
        def help_node(state):
            """Help agent node."""
            query = state["query"]
            response = self.help_agent.process_message(query)
            return {"agent_responses": {**state.get("agent_responses", {}), "help": response}}
        
        def manage_node(state):
            """Manage agent node."""
            query = state["query"]
            response = self.manage_agent.process_message(query)
            return {"agent_responses": {**state.get("agent_responses", {}), "manage": response}}
        
        def marketing_node(state):
            """Marketing agent node."""
            query = state["query"]
            response = self.marketing_agent.process_message(query)
            return {"agent_responses": {**state.get("agent_responses", {}), "marketing": response}}
        
        # Summarizer node to prepare the final response
        def summarizer(state):
            """Summarize and prepare the final response."""
            current_agent = state["current_agent"]
            agent_responses = state["agent_responses"]
            
            # If we have a response from the current agent, use it
            if current_agent in agent_responses:
                final_response = agent_responses[current_agent]
            else:
                # Fallback to help agent or any available response
                final_response = agent_responses.get("help", 
                                 next(iter(agent_responses.values()), 
                                 "I'm sorry, but I couldn't process your request."))
                
            # Add the response to conversation history
            history = state.get("conversation_history", [])
            history.append({"role": "user", "content": state["query"]})
            history.append({"role": "assistant", "content": final_response})
            
            return {"final_response": final_response, "conversation_history": history}
        
        # Add nodes to graph
        graph.add_node("router", router)
        graph.add_node("sales", sales_node)
        graph.add_node("help", help_node)
        graph.add_node("manage", manage_node)
        graph.add_node("marketing", marketing_node)
        graph.add_node("summarizer", summarizer)
        
        # Define edges
        graph.add_edge("router", "sales")
        graph.add_edge("router", "help")
        graph.add_edge("router", "manage")
        graph.add_edge("router", "marketing")
        
        graph.add_conditional_edges(
            "router",
            lambda state: state["current_agent"]
        )
        
        graph.add_edge("sales", "summarizer")
        graph.add_edge("help", "summarizer")
        graph.add_edge("manage", "summarizer")
        graph.add_edge("marketing", "summarizer")
        graph.add_edge("summarizer", END)
        
        # Set the entry point
        graph.set_entry_point("router")
        
        return graph.compile()
    
    def process_query(self, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Process a user query through the agent graph.
        
        Args:
            query: The user's query
            conversation_history: Optional conversation history
            
        Returns:
            The final response from the appropriate agent
        """
        if conversation_history is None:
            conversation_history = []
            
        # Run the graph
        result = self.graph.invoke({
            "query": query,
            "agent_responses": {},
            "conversation_history": conversation_history
        })
        
        return result["final_response"], result["conversation_history"]

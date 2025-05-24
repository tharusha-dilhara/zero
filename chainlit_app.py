import chainlit as cl
import os
from dotenv import load_dotenv
from utils.llm_manager import GroqLLMManager
from utils.memory import QdrantMemory
from agents.sales_agent import SalesAgent
from agents.help_agent import HelpAgent
from agents.manage_agent import ManageAgent
from agents.marketing_agent import MarketingAgent
from langgraph_setup.agent_graph import AgentGraph

# Load environment variables
load_dotenv()

# Initialize components
llm_manager = GroqLLMManager(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL", "llama3-70b-8192")
)

memory = QdrantMemory(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    collection_name="agent_memory"
)

# Initialize agents
sales_agent = SalesAgent(llm_manager=llm_manager, memory=memory)
help_agent = HelpAgent(llm_manager=llm_manager, memory=memory)
manage_agent = ManageAgent(llm_manager=llm_manager, memory=memory)
marketing_agent = MarketingAgent(llm_manager=llm_manager, memory=memory)

# Initialize agent graph
agent_graph = AgentGraph(llm_manager=llm_manager, memory=memory)

# Create a mapping of agents for UI selection
agent_mapping = {
    "sales": {
        "name": "Sales Agent", 
        "description": "For course information and enrollment assistance",
        "agent": sales_agent
    },
    "help": {
        "name": "Help Agent", 
        "description": "For technical support and general assistance",
        "agent": help_agent
    },
    "manage": {
        "name": "Manage Agent", 
        "description": "For operational and administrative questions",
        "agent": manage_agent
    },
    "marketing": {
        "name": "Marketing Agent", 
        "description": "For information about promotions and marketing",
        "agent": marketing_agent
    },
    "auto": {
        "name": "Auto-select", 
        "description": "Let the system choose the best agent for your query",
        "agent": None
    }
}

@cl.on_chat_start
async def on_chat_start():
    # Store user session information
    cl.user_session.set("history", [])
    cl.user_session.set("current_agent", "auto")
    
    # Create agent selection elements
    await cl.Message(
        content="Welcome to our Software Engineering Education Platform! How can I assist you today?"
    ).send()
    
    # Create agent selection elements
    actions = [
        cl.Action(
            name=agent_id,
            value=agent_id,
            description=info["description"],
            label=info["name"]
        )
        for agent_id, info in agent_mapping.items()
    ]
    
    await cl.Message(
        content="Choose which agent you'd like to talk to:",
        actions=actions
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Get user session data
    history = cl.user_session.get("history", [])
    current_agent = cl.user_session.get("current_agent", "auto")
    
    # Process the message based on current agent selection
    if current_agent == "auto":
        # Use the agent graph for automatic routing
        final_response, updated_history = agent_graph.process_query(message.content, history)
        cl.user_session.set("history", updated_history)
        
        await cl.Message(content=final_response).send()
    else:
        # Use the specifically selected agent
        agent = agent_mapping[current_agent]["agent"]
        response = agent.process_message(message.content)
        
        # Update history
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": response})
        cl.user_session.set("history", history)
        
        await cl.Message(content=response).send()

@cl.on_action
async def on_action(action):
    # Update the selected agent
    selected_agent = action.name
    cl.user_session.set("current_agent", selected_agent)
    
    agent_name = agent_mapping[selected_agent]["name"]
    
    if selected_agent == "auto":
        await cl.Message(f"You're now chatting with our automated agent router. It will select the best agent to answer your questions.").send()
    else:
        await cl.Message(f"You're now chatting with our {agent_name}. How can I help you today?").send()

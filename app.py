from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from agents.sales_agent import SalesAgent
from agents.help_agent import HelpAgent
from agents.manage_agent import ManageAgent
from agents.marketing_agent import MarketingAgent
from utils.memory import QdrantMemory
from utils.llm_manager import GroqLLMManager

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize dependencies
qdrant_memory = QdrantMemory(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    collection_name="agent_memory"
)

llm_manager = GroqLLMManager(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL", "llama3-70b-8192")
)

# Initialize agents
sales_agent = SalesAgent(llm_manager=llm_manager, memory=qdrant_memory)
help_agent = HelpAgent(llm_manager=llm_manager, memory=qdrant_memory)
manage_agent = ManageAgent(llm_manager=llm_manager, memory=qdrant_memory)
marketing_agent = MarketingAgent(llm_manager=llm_manager, memory=qdrant_memory)

# Map agent IDs to agent instances
agents = {
    "sales": sales_agent,
    "help": help_agent,
    "manage": manage_agent,
    "marketing": marketing_agent
}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    user_id = data.get('user_id')
    
    # Determine which agent should handle the request
    agent_id = data.get('agent_id', 'help')  # Default to help agent
    
    if agent_id not in agents:
        return jsonify({"error": "Invalid agent ID"}), 400
    
    # Process the message with the selected agent
    response = agents[agent_id].process_message(user_input, user_id)
    
    return jsonify({"response": response})

@app.route('/agent/<agent_id>', methods=['POST'])
def agent_endpoint(agent_id):
    if agent_id not in agents:
        return jsonify({"error": "Agent not found"}), 404
    
    data = request.json
    user_input = data.get('message')
    user_id = data.get('user_id')
    
    # Process the message with the specified agent
    response = agents[agent_id].process_message(user_input, user_id)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))

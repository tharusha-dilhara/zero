from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from utils.memory import QdrantMemory
from utils.llm_manager import GroqLLMManager
import uuid

class BaseAgent(ABC):
    def __init__(self, 
                 llm_manager: GroqLLMManager, 
                 memory: QdrantMemory,
                 system_message: str = None,
                 name: str = "Agent",
                 description: str = "A helpful AI agent"):
        """
        Initialize the base agent.
        
        Args:
            llm_manager: The LLM manager for generating responses
            memory: Vector memory for persistent storage
            system_message: System message to guide the agent's behavior
            name: Name of the agent
            description: Description of the agent's purpose
        """
        self.llm_manager = llm_manager
        self.memory = memory
        self.system_message = system_message
        self.name = name
        self.description = description
        self.conversation_history = {}
    
    def process_message(self, message: str, user_id: str = None) -> str:
        """
        Process a message from a user and return a response.
        
        Args:
            message: The user's message
            user_id: Unique identifier for the user
            
        Returns:
            Agent's response
        """
        # Create a user ID if not provided
        if user_id is None:
            user_id = str(uuid.uuid4())
            
        # Initialize conversation history for new users
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
            
        # Retrieve relevant memories
        relevant_memories = self.memory.search_memories(message, user_id=user_id)
        context = self._format_memories(relevant_memories)
        
        # Prepare messages for the LLM
        messages = self._prepare_messages(message, user_id, context)
        
        # Generate response
        response = self.llm_manager.generate_with_history(messages)
        
        # Update conversation history
        self.conversation_history[user_id].append({
            "role": "user",
            "content": message
        })
        self.conversation_history[user_id].append({
            "role": "assistant",
            "content": response
        })
        
        # Store interaction in memory
        self.memory.add_memory(
            text=f"User: {message}\nAgent: {response}",
            metadata={"interaction_type": "conversation"},
            user_id=user_id
        )
        
        return response
    
    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories as context for the LLM."""
        if not memories:
            return ""
            
        formatted_memories = "Previous relevant interactions:\n"
        for i, memory in enumerate(memories, 1):
            formatted_memories += f"{i}. {memory['text']}\n"
            
        return formatted_memories
    
    def _prepare_messages(self, message: str, user_id: str, context: str) -> List[Dict[str, str]]:
        """Prepare messages for the LLM including history and context."""
        messages = []
        
        # Add system message if available
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
            
        # Add abbreviated conversation history (last 5 messages)
        if user_id in self.conversation_history:
            history = self.conversation_history[user_id][-10:]  # Last 5 exchanges (10 messages)
            messages.extend(history)
            
        # Add context from memory if available
        if context:
            messages.append({
                "role": "system",
                "content": f"Additional context: {context}"
            })
            
        # Add the current user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        return messages
    
    @abstractmethod
    def get_agent_prompt(self) -> str:
        """Return the agent-specific system prompt."""
        pass

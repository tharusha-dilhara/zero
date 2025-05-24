import os
import groq
from typing import List, Dict, Optional, Any, Union

class GroqLLMManager:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-70b-8192"):
        """
        Initialize the Groq LLM Manager.
        
        Args:
            api_key: Groq API key (will use environment variable if not provided)
            model: The model name to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        self.model = model
        self.client = groq.Client(api_key=self.api_key)
    
    def generate(self, 
                prompt: str, 
                system_message: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 1024) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message to guide the model
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
            
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_with_history(self,
                             messages: List[Dict[str, str]],
                             temperature: float = 0.7,
                             max_tokens: int = 1024) -> str:
        """
        Generate a response with conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content

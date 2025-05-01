import os
from typing import List, Dict
import openai
from dotenv import load_dotenv

load_dotenv()

class LLMEngine:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("MODEL_NAME", "gpt-4")
        self.max_tokens = int(os.getenv("MAX_TOKENS", 2000))
        self.temperature = float(os.getenv("TEMPERATURE", 0.7))
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = self.api_key

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: Generated response
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response['choices'][0]['message']['content']

    def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for a given text using OpenAI's embedding model.
        
        Args:
            text: Input text to get embeddings for
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
            )
            return response['data'][0]['embedding']
        except Exception as e:
            raise Exception(f"Error getting embeddings: {str(e)}") 
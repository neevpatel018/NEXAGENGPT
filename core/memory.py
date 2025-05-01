from typing import List, Dict
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory

class ChatMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.max_history = 10  # Maximum number of conversation turns to keep

    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        if role == "user":
            self.memory.chat_memory.add_user_message(content)
        else:
            self.memory.chat_memory.add_ai_message(content)

    def get_history(self) -> List[Dict]:
        """
        Get the conversation history.
        
        Returns:
            List[Dict]: List of messages with 'role' and 'content' keys
        """
        history = self.memory.chat_memory.messages
        return [
            {"role": msg.type, "content": msg.content}
            for msg in history[-self.max_history:]
        ]

    def clear(self):
        """Clear the conversation history."""
        self.memory.clear() 
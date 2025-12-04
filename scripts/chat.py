import json
import time
import asyncio
from copy import deepcopy
from typing import List, Dict, Any, Optional
from uuid import uuid4

from scripts.llm import llm_call, llm_response


class ChatSession:
    """
    Manages a chat session with an LLM.
    
    This class:
    - Maintains conversation history
    - Handles async chat interactions
    - Supports session copying and branching
    """
    
    def __init__(
        self,
        model: str,
        temperature: float,
        config: Any,
        messages: Optional[List[Dict]] = None
    ) -> None:
        """Initialize a chat session."""
        self.model = model
        self.temperature = temperature
        self.config = config
        self.messages: List[Dict[str, str]] = [] if messages is None else deepcopy(messages)
        self._lock = False
        self.session_id = str(uuid4())
    
    def copy(self) -> "ChatSession":
        """Create a copy of this chat session."""
        return ChatSession(
            self.model,
            self.temperature,
            self.config,
            self.messages
        )
    
    def set_system(self, system: str) -> None:
        """Set the system message for this session."""
        self.messages.append({
            "role": "system",
            "content": system
        })
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content
        })
    
    async def chat(self, content: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            content: The user message content
            
        Returns:
            The assistant's response
        """
        if self._lock:
            raise RuntimeError("Chat session is locked (concurrent call in progress)")
        
        self._lock = True
        
        try:
            self.messages.append({
                "role": "user",
                "content": content
            })
            
            uid = llm_call(
                self.model,
                self.temperature,
                self.messages,
                self.config
            )
            
            response = await llm_response(uid, self.config)
            
            self.messages.append({
                "role": "assistant",
                "content": response
            })
            
            return response
        finally:
            self._lock = False
    
    def chat_sync(self, content: str) -> str:
        """Synchronous version of chat."""
        return asyncio.run(self.chat(content))
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return deepcopy(self.messages)
    
    def clear_history(self, keep_system: bool = True) -> None:
        """Clear the conversation history."""
        if keep_system:
            system_messages = [m for m in self.messages if m["role"] == "system"]
            self.messages = system_messages
        else:
            self.messages = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "model": self.model,
            "temperature": self.temperature,
            "messages": self.messages,
            "message_count": len(self.messages)
        }


class MultiChatSession:
    """
    Manages multiple parallel chat sessions.
    
    Useful for running multiple LLM calls concurrently.
    """
    
    def __init__(
        self,
        model: str,
        temperature: float,
        config: Any,
        n_sessions: int = 1
    ) -> None:
        """Initialize multiple chat sessions."""
        self.sessions = [
            ChatSession(model, temperature, config)
            for _ in range(n_sessions)
        ]
        self.model = model
        self.temperature = temperature
        self.config = config
    
    def set_system_all(self, system: str) -> None:
        """Set the same system message for all sessions."""
        for session in self.sessions:
            session.set_system(system)
    
    async def chat_parallel(self, contents: List[str]) -> List[str]:
        """
        Send messages to all sessions in parallel.
        
        Args:
            contents: List of messages (one per session)
            
        Returns:
            List of responses
        """
        if len(contents) != len(self.sessions):
            raise ValueError("Number of contents must match number of sessions")
        
        tasks = [
            session.chat(content)
            for session, content in zip(self.sessions, contents)
        ]
        
        return await asyncio.gather(*tasks)
    
    async def broadcast_chat(self, content: str) -> List[str]:
        """
        Send the same message to all sessions.
        
        Args:
            content: The message to broadcast
            
        Returns:
            List of responses from all sessions
        """
        tasks = [session.chat(content) for session in self.sessions]
        return await asyncio.gather(*tasks)


import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from uuid import uuid4


def llm_call(
    model: str,
    temperature: float,
    messages: List[Dict[str, str]],
    config: Any
) -> str:
    """
    Submit an LLM call to the task queue.
    
    Args:
        model: The model to use
        temperature: Sampling temperature
        messages: The conversation messages
        config: Configuration object
        
    Returns:
        Task ID for retrieving the result
    """
    task_id = str(uuid4())
    
    task_data = {
        "task_id": task_id,
        "model": model,
        "temperature": temperature,
        "messages": messages
    }
    
    r = config.get_redis_connect()
    r.lpush(config.REDIS_GPT_TASK_KEY, json.dumps(task_data))
    
    return task_id


async def llm_response(task_id: str, config: Any, timeout: int = 120) -> str:
    """
    Wait for and retrieve an LLM response.
    
    Args:
        task_id: The task ID to wait for
        config: Configuration object
        timeout: Maximum wait time in seconds
        
    Returns:
        The LLM response content
    """
    r = config.get_redis_connect()
    start_time = time.time()
    wait_time = config.LLM_CALL_WAITING_TIME
    
    while time.time() - start_time < timeout:
        result = r.hget(config.REDIS_GPT_RESULT_KEY, task_id)
        
        if result:
            result_data = json.loads(result)
            # Clean up
            r.hdel(config.REDIS_GPT_RESULT_KEY, task_id)
            return result_data.get("response", "")
        
        await asyncio.sleep(wait_time)
    
    raise TimeoutError(f"LLM response timeout after {timeout} seconds")


def llm_call_sync(
    model: str,
    temperature: float,
    messages: List[Dict[str, str]],
    config: Any,
    timeout: int = 120
) -> str:
    """
    Synchronous LLM call that waits for the response.
    
    Args:
        model: The model to use
        temperature: Sampling temperature
        messages: The conversation messages
        config: Configuration object
        timeout: Maximum wait time
        
    Returns:
        The LLM response content
    """
    task_id = llm_call(model, temperature, messages, config)
    return asyncio.run(llm_response(task_id, config, timeout))


class DirectLLMClient:
    """
    Direct LLM client that calls the API without Redis queue.
    
    Useful for development and testing.
    """
    
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        """Initialize the direct LLM client."""
        import os
        self.api_base_url = api_base_url or os.environ.get(
            'OPENAI_BASE_URL', 'https://api.openai.com/v1/chat/completions'
        )
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
    
    def call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Make a direct LLM API call."""
        import requests
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        response = requests.post(
            self.api_base_url,
            headers=headers,
            json=data,
            timeout=120
        )
        
        result = response.json()
        
        if "choices" not in result:
            raise ValueError(f"Invalid response: {result}")
        
        return result["choices"][0]["message"]["content"]
    
    async def call_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Async version of direct LLM call."""
        import aiohttp
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_base_url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                result = await response.json()
        
        if "choices" not in result:
            raise ValueError(f"Invalid response: {result}")
        
        return result["choices"][0]["message"]["content"]


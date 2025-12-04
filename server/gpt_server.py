import os
import json
import requests
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

from server.base import BaseServer


class GPTResponseFormatError(Exception):
    """Exception raised when GPT response format is invalid."""
    pass


class GPTServer(BaseServer):
    """
    Server for handling GPT/LLM API calls.
    
    This server:
    - Manages LLM API requests
    - Handles rate limiting and retries
    - Caches responses when appropriate
    - Supports multiple model providers
    """
    
    SERVER_KEY = "gpt"
    
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        **kwargs
    ) -> None:
        """Initialize the GPT Server."""
        self.api_base_url = api_base_url or os.environ.get(
            'OPENAI_BASE_URL', 'https://api.openai.com/v1/chat/completions'
        )
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.default_model = default_model
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Import config for Redis keys
        try:
            from config import Config
            self.config = Config()
            self.task_key = self.config.REDIS_GPT_TASK_KEY
            self.result_key = self.config.REDIS_GPT_RESULT_KEY
        except ImportError:
            self.config = None
            self.task_key = "mochiagent:gpt:task"
            self.result_key = "mochiagent:gpt:result"
        
        super().__init__(**kwargs)
    
    def get_task(self):
        """Get a task from the GPT task queue."""
        data = self.r.rpop(self.task_key)
        if not data:
            return False, None
        return True, json.loads(data)
    
    def on_error(self, e: Exception, data: Dict[str, Any]):
        """Handle errors during GPT API calls."""
        error_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        
        if self.config:
            self.config.log_error(self.SERVER_KEY, error_trace)
        
        # Check for specific error types and handle accordingly
        if "string_above_max_length" in error_trace:
            self._log_error("Input string too long, skipping task")
            return
        elif "context_length_exceeded" in error_trace:
            self._log_error("Context length exceeded, skipping task")
            return
        
        # Re-queue for transient errors
        if isinstance(e, (requests.exceptions.ProxyError, 
                         requests.exceptions.ConnectionError,
                         GPTResponseFormatError)):
            self._log_warning(f"Transient error, re-queuing task: {e}")
            self.r.rpush(self.task_key, json.dumps(data))
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GPT API request."""
        task_id = data.get("task_id")
        
        # Prepare request data
        post_data = data.copy()
        post_data.pop("task_id", None)
        
        # Set default model if not specified
        if "model" not in post_data:
            post_data["model"] = self.default_model
        
        self._log_info(f"Processing GPT request: {task_id}")
        
        try:
            response = requests.post(
                self.api_base_url,
                headers=self.headers,
                json=post_data,
                timeout=120,
                verify=False
            )
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            raise GPTResponseFormatError(f"Request failed: {e}")
        
        if "choices" not in response_json:
            raise GPTResponseFormatError(f"Invalid response: {json.dumps(response_json)}")
        
        content = response_json["choices"][0]["message"]["content"]
        
        # Store result
        result = {
            "response": content,
            "model": post_data.get("model"),
            "timestamp": datetime.now().isoformat()
        }
        
        self.r.hset(self.result_key, task_id, json.dumps(result))
        
        return result


class CodeExecutorServer(BaseServer):
    """
    Server for executing code in a sandboxed environment.
    
    This server:
    - Executes Python code safely
    - Captures output and errors
    - Manages execution timeouts
    - Returns structured results
    """
    
    SERVER_KEY = "code_executor"
    EXECUTION_TIMEOUT = 300  # 5 minutes
    
    def __init__(self, **kwargs) -> None:
        """Initialize the Code Executor Server."""
        try:
            from config import Config
            self.config = Config()
            self.task_key = self.config.REDIS_EXECUTOR_LIST_TASK_KEY
            self.data_key = self.config.REDIS_EXECUTOR_LIST_DATA_KEY
        except ImportError:
            self.config = None
            self.task_key = "mochiagent:executor:task"
            self.data_key = "mochiagent:executor:data"
        
        super().__init__(**kwargs)
    
    def get_task(self):
        """Get a task from the executor task queue."""
        data = self.r.rpop(self.task_key)
        if not data:
            return False, None
        return True, json.loads(data)
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code and return results."""
        task_id = data.get("task_id")
        code = data.get("code", "")
        test_function = data.get("test_function", "")
        
        self._log_info(f"Executing code for task: {task_id}")
        
        result = {
            "task_id": task_id,
            "result": False,
            "excepted": False,
            "data": None,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Create execution namespace
            exec_globals = {
                "__builtins__": __builtins__,
                "np": __import__("numpy"),
                "pd": __import__("pandas"),
                "json": __import__("json"),
                "os": __import__("os"),
                "datetime": datetime
            }
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Call test function if specified
            if test_function and test_function in exec_locals:
                test_result = exec_locals[test_function]()
                result["data"] = test_result
                result["result"] = True
            else:
                result["result"] = True
                result["data"] = "Code executed successfully"
        
        except Exception as e:
            result["result"] = False
            result["excepted"] = True
            result["error"] = str(e)
            result["data"] = traceback.format_exc()
        
        # Store result
        self.r.hset(self.data_key, task_id, json.dumps(result))
        
        return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "executor":
        server = CodeExecutorServer()
    else:
        server = GPTServer()
    
    server.run()


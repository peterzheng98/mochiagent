import json
import traceback
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
from datetime import datetime

from server.base import BaseServer


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
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize the Code Executor Server."""
        self.working_dir = working_dir or tempfile.gettempdir()
        
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
    
    def _execute_in_subprocess(
        self,
        code: str,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Execute code in a subprocess for isolation."""
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            dir=self.working_dir,
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute in subprocess
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _execute_inline(
        self,
        code: str,
        test_function: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute code inline (less isolated but faster)."""
        exec_globals = {
            "__builtins__": __builtins__,
        }
        
        # Try to import common libraries
        try:
            exec_globals["np"] = __import__("numpy")
        except ImportError:
            pass
        try:
            exec_globals["pd"] = __import__("pandas")
        except ImportError:
            pass
        
        exec_globals["json"] = __import__("json")
        exec_globals["os"] = __import__("os")
        exec_globals["datetime"] = datetime
        
        exec_locals = {}
        
        try:
            exec(code, exec_globals, exec_locals)
            
            # Call test function if specified
            if test_function and test_function in exec_locals:
                test_result = exec_locals[test_function]()
                return {
                    "success": True,
                    "result": test_result,
                    "error": None
                }
            else:
                return {
                    "success": True,
                    "result": "Code executed successfully",
                    "error": None
                }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code and return results."""
        task_id = data.get("task_id")
        code = data.get("code", "")
        test_function = data.get("test_function")
        use_subprocess = data.get("use_subprocess", False)
        timeout = data.get("timeout", self.EXECUTION_TIMEOUT)
        
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
            if use_subprocess:
                exec_result = self._execute_in_subprocess(code, timeout)
                result["result"] = exec_result["success"]
                result["data"] = exec_result.get("stdout", "")
                if not exec_result["success"]:
                    result["error"] = exec_result.get("stderr", "")
                    result["excepted"] = True
            else:
                exec_result = self._execute_inline(code, test_function)
                result["result"] = exec_result["success"]
                result["data"] = exec_result.get("result")
                if not exec_result["success"]:
                    result["error"] = exec_result.get("error", "")
                    result["excepted"] = True
                    if "traceback" in exec_result:
                        result["data"] = exec_result["traceback"]
        
        except Exception as e:
            result["result"] = False
            result["excepted"] = True
            result["error"] = str(e)
            result["data"] = traceback.format_exc()
        
        # Store result
        self.r.hset(self.data_key, task_id, json.dumps(result))
        
        return result


if __name__ == "__main__":
    server = CodeExecutorServer()
    server.run()


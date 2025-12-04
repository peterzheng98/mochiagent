import os
import json
import time
import shutil
from typing import Any, Dict, List, Optional

from config import Config
from utils import generate_task_id, yprint


class Status:
    """
    Tracks the status and state of a task throughout its lifecycle.
    
    This class:
    - Stores intermediate results
    - Tracks progress through pipeline stages
    - Provides state persistence via Redis
    """
    
    def __init__(
        self,
        task_id: str,
        config: Config,
        update: bool = True
    ) -> None:
        """Initialize status tracker."""
        self.data: Dict[str, Any] = {}
        self.task_id = task_id
        self.config = config
        self.update = update
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to track all attributes in data dict."""
        super().__setattr__(name, value)
        if name not in ["data", "task_id", "config", "update"]:
            if hasattr(self, "data"):
                self.data[name] = value
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get a value from the status data."""
        return self.data.get(name, default)
    
    def set(self, name: str, value: Any) -> None:
        """Set a value in the status data."""
        self.data[name] = value
        setattr(self, name, value)
    
    def show(self) -> None:
        """Print the current status data."""
        yprint("=" * 30)
        print(json.dumps(self.data, indent=2, default=str))
        yprint("=" * 30)
    
    def save(self, output_path: str = "output.json") -> None:
        """Save status data to a JSON file."""
        with open(output_path, "w", encoding="utf8") as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def status_update(self, stage: str) -> None:
        """Update status in Redis."""
        if not self.update:
            return
        
        now = int(time.time())
        self.upload_status = {
            "stage": stage,
            "time": now
        }
        
        r = self.config.get_redis_connect()
        
        # Push to status list
        r.lpush(
            f"{self.config.REDIS_STATUS_LIST_KEY}:{self.task_id}",
            json.dumps({
                "uid": generate_task_id(),
                "stage": stage,
                "timestamp": now
            })
        )
        
        # Update status data hash
        r.hset(
            self.config.REDIS_STATUS_DATA_KEY,
            self.task_id,
            json.dumps(self.data, default=str)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return self.data.copy()


class Task:
    """
    Represents a task to be processed by the agent.
    
    This class:
    - Manages input data (EHR, lab tests, files)
    - Tracks task status and results
    - Handles file management
    """
    
    def __init__(
        self,
        info: Dict[str, Any],
        config: Config,
        update: bool = False,
        official: bool = False
    ) -> None:
        """
        Initialize a task.
        
        Args:
            info: Task information containing:
                - ehr: List of EHR records
                - lab_tests: 2D list of lab test values
                - files: Optional list of file references
            config: Configuration object
            update: Whether to push status updates to Redis
            official: Whether this is an official (production) task
        """
        self.official = official
        self.config = config
        
        # Extract task data
        self.ehr = info.get("ehr", [])
        self.lab_tests = info.get("lab_tests", [])
        self.question = info.get("question", "")
        self.files = info.get("files", [])
        
        # Initialize status
        self.status = Status(self.config.get_task(), config, update)
        self.tool_manager = ToolManager(config)
        
        # Prepare task
        self._prepare()
    
    def _prepare(self) -> None:
        """Prepare task directories and initial state."""
        task_id = self.config.get_task()
        task_path = os.path.join(
            self.config.TASK_DIR,
            self.config.time_path,
            task_id
        )
        
        self.config.task_path = task_path
        self.status.task_path = task_path
        self.status.task_id = task_id
        self.status.ehr = self.ehr
        self.status.lab_tests = self.lab_tests
        self.status.raw_question = self.question
        self.status.backtrack = []
        self.status.files = self.file_list
        
        # Create task directory
        os.makedirs(task_path, exist_ok=True)
        
        # Copy input files to task directory
        for file_info in self.files:
            if isinstance(file_info, dict):
                src_path = file_info.get("path", "")
                if os.path.exists(src_path):
                    shutil.copy(src_path, task_path)
        
        # Initialize tool status
        self.status.tools = {}
        self.status.rescored = False
        self.status.code = {}
    
    @property
    def file_list(self) -> List[str]:
        """Get list of file names."""
        result = []
        for file_info in self.files:
            if isinstance(file_info, dict):
                result.append(file_info.get("name", ""))
            elif isinstance(file_info, str):
                result.append(file_info)
        return result
    
    def backtrack_update(self) -> None:
        """Save current state to backtrack history and reset."""
        self.status.backtrack.append({
            "workflow": self.status.get("workflow"),
            "tool_used": self.status.get("tool_used"),
            "workflow_stages": self.status.get("workflow_stages"),
            "resource_pool": self.status.get("resource_pool"),
            "code_result": self.status.get("code_result"),
            "code": self.status.get("code")
        })
        
        # Reset current state
        self.status.tool_used = {}
        self.status.workflow_stages = []
        self.status.resource_pool = []
        self.status.code_result = {}
        self.status.code = {}
    
    def get_input_data(self) -> Dict[str, Any]:
        """Get the input data for processing."""
        return {
            "ehr": self.ehr,
            "lab_tests": self.lab_tests,
            "question": self.question,
            "files": self.files
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.config.get_task(),
            "ehr": self.ehr,
            "lab_tests": self.lab_tests,
            "question": self.question,
            "files": self.file_list,
            "status": self.status.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Config) -> "Task":
        """Create a task from a dictionary."""
        return cls(data, config)
    
    @classmethod
    def fake_task(cls, config: Config) -> "Task":
        """Create a fake task for testing."""
        fake_info = {
            "ehr": [],
            "lab_tests": [],
            "question": "",
            "files": []
        }
        return Task(fake_info, config)


class ToolManager:
    """
    Manages tool registration and retrieval.
    
    This class:
    - Tracks available tools
    - Retrieves tool documentation
    - Manages tool activation status
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize the tool manager."""
        self.config = config
        self.r = config.get_redis_connect()
    
    def get_tool_list(self, ensure_active: bool = True) -> List[str]:
        """Get list of available tools."""
        if ensure_active:
            tools = self.r.hkeys(self.config.REDIS_ACTIVE_TOOL_KEY)
            return list(tools) if tools else []
        else:
            tool_ids = self.r.hkeys(self.config.REDIS_TOOL_INFO_KEY)
            return list(tool_ids) if tool_ids else []
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        tool_id = self.r.hget(self.config.REDIS_ACTIVE_TOOL_KEY, tool_name)
        if not tool_id:
            return None
        
        tool_info = self.r.hget(self.config.REDIS_TOOL_INFO_KEY, tool_id)
        if not tool_info:
            return None
        
        return json.loads(tool_info)
    
    def get_tool_doc(self, tool_name: str) -> Optional[str]:
        """Get documentation for a tool."""
        info = self.get_tool_info(tool_name)
        if info:
            return info.get("tool_doc")
        
        # Try to load from file
        doc_path = os.path.join(self.config.TOOL_DOC_DIR, tool_name)
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf8") as f:
                return f.read()
        
        return None
    
    def get_tool_code(self, tool_name: str) -> Optional[str]:
        """Get code for a tool."""
        info = self.get_tool_info(tool_name)
        if info:
            return info.get("tool_code")
        
        # Try to load from file
        code_path = os.path.join(self.config.TOOL_CODE_DIR, tool_name)
        if os.path.exists(code_path):
            with open(code_path, "r", encoding="utf8") as f:
                return f.read()
        
        return None
    
    def register_tool(
        self,
        tool_name: str,
        tool_doc: str,
        tool_code: str
    ) -> str:
        """Register a new tool."""
        tool_id = generate_task_id()
        
        tool_info = {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "tool_doc": tool_doc,
            "tool_code": tool_code
        }
        
        self.r.hset(self.config.REDIS_TOOL_INFO_KEY, tool_id, json.dumps(tool_info))
        self.r.hset(self.config.REDIS_ACTIVE_TOOL_KEY, tool_name, tool_id)
        
        return tool_id
    
    def deactivate_tool(self, tool_name: str) -> bool:
        """Deactivate a tool."""
        result = self.r.hdel(self.config.REDIS_ACTIVE_TOOL_KEY, tool_name)
        return bool(result)


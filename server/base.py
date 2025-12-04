import json
import time
import redis
import asyncio
import traceback
from abc import ABC, abstractmethod
from uuid import uuid4
from typing import Any, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ServerStatus(Enum):
    """Enumeration of server status states."""
    INITIALIZING = "initializing"
    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ServerMetrics:
    """Metrics tracking for server performance."""
    tasks_processed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    last_task_time: Optional[datetime] = None
    uptime_start: datetime = field(default_factory=datetime.now)
    
    def record_task(self, duration: float, success: bool = True) -> None:
        """Record metrics for a completed task."""
        if success:
            self.tasks_processed += 1
        else:
            self.tasks_failed += 1
        self.total_processing_time += duration
        total_tasks = self.tasks_processed + self.tasks_failed
        self.average_processing_time = self.total_processing_time / total_tasks
        self.last_task_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
            "last_task_time": self.last_task_time.isoformat() if self.last_task_time else None,
            "uptime_seconds": (datetime.now() - self.uptime_start).total_seconds()
        }


class BaseServer(ABC):
    """
    Base server class providing common functionality for all MCP servers.
    
    This class handles:
    - Redis connection management
    - Server registration and heartbeat
    - Task queue polling
    - Error handling and recovery
    - Metrics collection
    """
    
    CENTER_KEY_PREFIX: str = "mochiagent:server_center"
    HEARTBEAT_KEY: str = "mochiagent:server_heartbeat"
    DEAD_POOL_KEY: str = "mochiagent:server_dead"
    SERVER_KEY: str = "base"
    SLEEP_TIME: float = 0.5
    HEARTBEAT_INTERVAL: int = 5
    MAX_CONSECUTIVE_ERRORS: int = 10
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None
    ) -> None:
        """Initialize the base server."""
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        
        self.r = self._create_redis_connection()
        self.center = f"{self.CENTER_KEY_PREFIX}:{self.SERVER_KEY}"
        self.id = str(uuid4())
        self.status = ServerStatus.INITIALIZING
        self.metrics = ServerMetrics()
        self.consecutive_errors = 0
        self._shutdown_flag = False
        self._last_heartbeat = time.time()
        
        self._register()
    
    def _create_redis_connection(self) -> redis.Redis:
        """Create a Redis connection."""
        return redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True
        )
    
    def _register(self) -> None:
        """Register the server in the Redis server center."""
        server_info = {
            "id": self.id,
            "status": self.status.value,
            "server_type": self.SERVER_KEY,
            "registered_at": datetime.now().isoformat(),
            "host": self.redis_host,
            "metrics": self.metrics.to_dict()
        }
        self.r.hset(self.center, self.id, json.dumps(server_info))
        self.r.hset(self.HEARTBEAT_KEY, self.id, int(time.time()))
        self._log_info(f"Server registered with ID: {self.id}")
    
    def _update_status(self, status: ServerStatus) -> None:
        """Update the server status in Redis."""
        self.status = status
        try:
            info = self.r.hget(self.center, self.id)
            if info:
                info = json.loads(info)
                info["status"] = status.value
                info["metrics"] = self.metrics.to_dict()
                info["last_updated"] = datetime.now().isoformat()
                self.r.hset(self.center, self.id, json.dumps(info))
        except Exception as e:
            self._log_error(f"Failed to update status: {e}")
    
    def _heartbeat(self) -> None:
        """Send heartbeat to Redis."""
        current_time = time.time()
        if current_time - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.r.hset(self.HEARTBEAT_KEY, self.id, int(current_time))
            self._last_heartbeat = current_time
    
    def _check_alive(self) -> bool:
        """Check if the server should continue running."""
        if self._shutdown_flag:
            return False
        is_dead = self.r.hget(self.DEAD_POOL_KEY, self.id)
        return not bool(is_dead)
    
    def _log_info(self, message: str) -> None:
        """Log an info message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{self.SERVER_KEY}:{self.id[:8]}] INFO: {message}")
    
    def _log_error(self, message: str) -> None:
        """Log an error message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{self.SERVER_KEY}:{self.id[:8]}] ERROR: {message}")
    
    def _log_warning(self, message: str) -> None:
        """Log a warning message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{self.SERVER_KEY}:{self.id[:8]}] WARNING: {message}")
    
    @abstractmethod
    def get_task(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Get a task from the task queue.
        
        Returns:
            Tuple of (has_task, task_data)
        """
        pass
    
    @abstractmethod
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            data: Task data
            
        Returns:
            Result data
        """
        pass
    
    def on_error(self, error: Exception, data: Dict[str, Any]) -> None:
        """
        Handle an error that occurred during task execution.
        
        Override this method to implement custom error handling.
        """
        self._log_error(f"Task execution failed: {error}")
        self._log_error(traceback.format_exc())
        self.consecutive_errors += 1
        
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self._log_error("Too many consecutive errors, pausing server")
            self._update_status(ServerStatus.ERROR)
            time.sleep(30)  # Pause before resuming
            self.consecutive_errors = 0
    
    def on_success(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Handle successful task completion.
        
        Override this method to implement custom success handling.
        """
        self.consecutive_errors = 0
    
    def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        self._log_info("Initiating shutdown...")
        self._shutdown_flag = True
        self._update_status(ServerStatus.SHUTDOWN)
        self.r.hdel(self.center, self.id)
        self.r.hdel(self.HEARTBEAT_KEY, self.id)
        self._log_info("Server shutdown complete")
    
    def run(self) -> None:
        """Main server loop."""
        self._log_info("Server starting...")
        self._update_status(ServerStatus.WAITING)
        
        try:
            while self._check_alive():
                self._heartbeat()
                
                has_task, data = self.get_task()
                if not has_task or data is None:
                    time.sleep(self.SLEEP_TIME)
                    continue
                
                self._update_status(ServerStatus.RUNNING)
                start_time = time.time()
                
                try:
                    result = self.execute(data)
                    duration = time.time() - start_time
                    self.metrics.record_task(duration, success=True)
                    self.on_success(data, result)
                except Exception as e:
                    duration = time.time() - start_time
                    self.metrics.record_task(duration, success=False)
                    self.on_error(e, data)
                
                self._update_status(ServerStatus.WAITING)
        
        except KeyboardInterrupt:
            self._log_info("Received interrupt signal")
        finally:
            self.shutdown()


class MCPServer(BaseServer):
    """
    Model Context Protocol (MCP) Server base class.
    
    This extends BaseServer with MCP-specific functionality including:
    - Protocol message handling
    - Tool registration
    - Resource management
    - Prompt templates
    """
    
    PROTOCOL_VERSION: str = "1.0"
    
    def __init__(
        self,
        task_queue_key: str,
        result_queue_key: str,
        **kwargs
    ) -> None:
        """Initialize the MCP server."""
        self.task_queue_key = task_queue_key
        self.result_queue_key = result_queue_key
        self._tools: Dict[str, Callable] = {}
        self._resources: Dict[str, Any] = {}
        self._prompts: Dict[str, str] = {}
        
        super().__init__(**kwargs)
        
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self) -> None:
        """Register available tools. Override in subclasses."""
        pass
    
    def _register_resources(self) -> None:
        """Register available resources. Override in subclasses."""
        pass
    
    def _register_prompts(self) -> None:
        """Register prompt templates. Override in subclasses."""
        pass
    
    def register_tool(self, name: str, handler: Callable, description: str = "") -> None:
        """Register a tool handler."""
        self._tools[name] = {
            "handler": handler,
            "description": description
        }
        self._log_info(f"Registered tool: {name}")
    
    def register_resource(self, uri: str, resource: Any) -> None:
        """Register a resource."""
        self._resources[uri] = resource
        self._log_info(f"Registered resource: {uri}")
    
    def register_prompt(self, name: str, template: str) -> None:
        """Register a prompt template."""
        self._prompts[name] = template
        self._log_info(f"Registered prompt: {name}")
    
    def get_task(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Get a task from the Redis task queue."""
        try:
            data = self.r.rpop(self.task_queue_key)
            if not data:
                return False, None
            return True, json.loads(data)
        except Exception as e:
            self._log_error(f"Failed to get task: {e}")
            return False, None
    
    def submit_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Submit a result to the Redis result queue."""
        try:
            result_data = {
                "task_id": task_id,
                "result": result,
                "server_id": self.id,
                "timestamp": datetime.now().isoformat(),
                "protocol_version": self.PROTOCOL_VERSION
            }
            self.r.hset(self.result_queue_key, task_id, json.dumps(result_data))
        except Exception as e:
            self._log_error(f"Failed to submit result: {e}")
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        handler = self._tools[tool_name]["handler"]
        return handler(**arguments)
    
    def get_resource(self, uri: str) -> Any:
        """Get a registered resource."""
        if uri not in self._resources:
            raise ValueError(f"Unknown resource: {uri}")
        return self._resources[uri]
    
    def get_prompt(self, name: str, **kwargs) -> str:
        """Get and format a prompt template."""
        if name not in self._prompts:
            raise ValueError(f"Unknown prompt: {name}")
        
        template = self._prompts[name]
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP task."""
        task_id = data.get("task_id")
        method = data.get("method")
        params = data.get("params", {})
        
        if not task_id:
            raise ValueError("Task ID is required")
        
        result = {}
        
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = {"content": self.call_tool(tool_name, arguments)}
        
        elif method == "resources/read":
            uri = params.get("uri")
            result = {"content": self.get_resource(uri)}
        
        elif method == "prompts/get":
            prompt_name = params.get("name")
            prompt_args = params.get("arguments", {})
            result = {"content": self.get_prompt(prompt_name, **prompt_args)}
        
        elif method == "tools/list":
            result = {"tools": list(self._tools.keys())}
        
        elif method == "resources/list":
            result = {"resources": list(self._resources.keys())}
        
        elif method == "prompts/list":
            result = {"prompts": list(self._prompts.keys())}
        
        else:
            result = self._handle_custom_method(method, params)
        
        self.submit_result(task_id, result)
        return result
    
    def _handle_custom_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle custom methods not covered by standard MCP protocol.
        Override in subclasses to implement custom methods.
        """
        raise ValueError(f"Unknown method: {method}")


class AsyncMCPServer(MCPServer):
    """Async version of MCP Server for handling concurrent requests."""
    
    def __init__(self, *args, max_concurrent_tasks: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_concurrent_tasks = max_concurrent_tasks
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    async def async_execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Async execution of tasks."""
        async with self._semaphore:
            return self.execute(data)
    
    async def async_run(self) -> None:
        """Async main server loop."""
        self._log_info("Async server starting...")
        self._update_status(ServerStatus.WAITING)
        
        try:
            while self._check_alive():
                self._heartbeat()
                
                has_task, data = self.get_task()
                if not has_task or data is None:
                    await asyncio.sleep(self.SLEEP_TIME)
                    continue
                
                task_id = data.get("task_id", str(uuid4()))
                task = asyncio.create_task(self._process_task(data))
                self._running_tasks[task_id] = task
                
                # Cleanup completed tasks
                completed = [
                    tid for tid, t in self._running_tasks.items()
                    if t.done()
                ]
                for tid in completed:
                    del self._running_tasks[tid]
        
        except asyncio.CancelledError:
            self._log_info("Server cancelled")
        finally:
            # Wait for all running tasks to complete
            if self._running_tasks:
                await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
            self.shutdown()
    
    async def _process_task(self, data: Dict[str, Any]) -> None:
        """Process a single task asynchronously."""
        start_time = time.time()
        
        try:
            result = await self.async_execute(data)
            duration = time.time() - start_time
            self.metrics.record_task(duration, success=True)
            self.on_success(data, result)
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_task(duration, success=False)
            self.on_error(e, data)


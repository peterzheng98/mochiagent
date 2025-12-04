import os
import redis
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


def time_dir() -> str:
    """Generate a time-based directory path."""
    now = datetime.now()
    return os.path.join(f"{now.year}-{now.month}", f"{now.day}")


@dataclass
class MCPServerConfig:
    """Configuration for individual MCP servers."""
    host: str = "localhost"
    port: int = 8080
    timeout: int = 30
    max_retries: int = 3
    heartbeat_interval: int = 5


@dataclass
class TransformerServerConfig(MCPServerConfig):
    """Configuration for transformer inference server."""
    port: int = 8081
    model_name: str = "transformer-ehr-v1"
    batch_size: int = 32
    max_sequence_length: int = 512
    device: str = "cpu"
    inference_timeout: int = 60


@dataclass
class WebSearchServerConfig(MCPServerConfig):
    """Configuration for web search server."""
    port: int = 8082
    search_engine: str = "duckduckgo"
    max_results: int = 10
    search_timeout: int = 15
    rate_limit_per_minute: int = 30


@dataclass
class TrajectoryServerConfig(MCPServerConfig):
    """Configuration for single-cell trajectory inference server."""
    port: int = 8083
    algorithm: str = "monocle3"
    n_neighbors: int = 15
    min_branch_length: int = 10
    root_selection_method: str = "automatic"


@dataclass
class ClusteringServerConfig(MCPServerConfig):
    """Configuration for clustering analysis server."""
    port: int = 8084
    algorithm: str = "leiden"
    resolution: float = 1.0
    n_clusters: Optional[int] = None
    random_state: int = 42


class Config:
    """Main configuration class for the Medical Agent system."""
    
    # LLM Configuration
    LLM_CALL_PASSWORD: str = "MochiAgent"
    LLM_CALL_WAITING_TIME: int = 1
    LLM_SERVER_IP: str = ""
    
    # Directory Configuration
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR: str = os.path.join(BASE_DIR, "log")
    TASK_DIR: str = os.path.join(BASE_DIR, "task")
    TOOL_DOC_DIR: str = os.path.join(BASE_DIR, "tool", "doc")
    TOOL_CODE_DIR: str = os.path.join(BASE_DIR, "tool", "code")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    
    # Logging Configuration
    SAVE_LOG: bool = True
    ECHO_INFO: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Memory Configuration
    SAVE_MEMORY: bool = False
    MEMORY_PREFIX: str = "v1.0.0"
    USE_MEMORY: bool = False
    USE_FILE_APPENDIX: bool = False
    
    # LLM Model Configuration
    BASE_LLM_MODEL: str = "gpt-4o-mini"
    SUPER_LLM_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Agent Configuration
    HIGHSCORE_TOOL_THRESHOLD: int = 5
    WORKFLOW_USED_TOOL_THRESHOLD: int = 5
    ACTION_RETRY_TIMES: int = 4
    MAX_WORKFLOW_ITERATIONS: int = 3
    
    # Executor Configuration
    EXECUTOR_CODE_WAITING_TIME: int = 1
    EXECUTOR_TIMEOUT: int = 300
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Redis Keys - Task Management
    REDIS_TASK_QUEUE_KEY: str = "mochiagent:task:queue"
    REDIS_TASK_RESULT_KEY: str = "mochiagent:task:result"
    REDIS_TASK_STATUS_KEY: str = "mochiagent:task:status"
    
    # Redis Keys - LLM Services
    REDIS_GPT_TASK_KEY: str = "mochiagent:gpt:task"
    REDIS_GPT_RESULT_KEY: str = "mochiagent:gpt:result"
    REDIS_LLAMA_TASK_KEY: str = "mochiagent:llama:task"
    REDIS_LLAMA_RESULT_KEY: str = "mochiagent:llama:result"
    
    # Redis Keys - MCP Servers
    REDIS_TRANSFORMER_TASK_KEY: str = "mochiagent:transformer:task"
    REDIS_TRANSFORMER_RESULT_KEY: str = "mochiagent:transformer:result"
    REDIS_WEBSEARCH_TASK_KEY: str = "mochiagent:websearch:task"
    REDIS_WEBSEARCH_RESULT_KEY: str = "mochiagent:websearch:result"
    REDIS_TRAJECTORY_TASK_KEY: str = "mochiagent:trajectory:task"
    REDIS_TRAJECTORY_RESULT_KEY: str = "mochiagent:trajectory:result"
    REDIS_CLUSTERING_TASK_KEY: str = "mochiagent:clustering:task"
    REDIS_CLUSTERING_RESULT_KEY: str = "mochiagent:clustering:result"
    
    # Redis Keys - Code Executor
    REDIS_EXECUTOR_LIST_TASK_KEY: str = "mochiagent:executor:task"
    REDIS_EXECUTOR_LIST_DATA_KEY: str = "mochiagent:executor:data"
    
    # Redis Keys - Tool Management
    REDIS_ACTIVE_TOOL_KEY: str = "mochiagent:tools:active"
    REDIS_TOOL_INFO_KEY: str = "mochiagent:tools:info"
    
    # Redis Keys - Status & Progress
    REDIS_STATUS_DATA_KEY: str = "mochiagent:status:data"
    REDIS_STATUS_LIST_KEY: str = "mochiagent:status:list"
    REDIS_PROGRESS_DATA_KEY: str = "mochiagent:progress:data"
    REDIS_RESULT_DATA_KEY: str = "mochiagent:result:data"
    
    # Redis Keys - Server Management
    REDIS_SERVER_CENTER_KEY: str = "mochiagent:server:center"
    REDIS_SERVER_HEARTBEAT_KEY: str = "mochiagent:server:heartbeat"
    REDIS_SERVER_DEAD_POOL_KEY: str = "mochiagent:server:dead"
    
    # Redis Keys - Memory & Knowledge Base
    REDIS_MEMORY_STORAGE_KEY: str = f"mochiagent:memory:{MEMORY_PREFIX}:storage"
    REDIS_MEMORY_INFO_KEY: str = "mochiagent:memory:info"
    REDIS_MEMORY_TASK_KEY: str = f"mochiagent:memory:{MEMORY_PREFIX}:task"
    REDIS_MEMORY_RESULT_KEY: str = f"mochiagent:memory:{MEMORY_PREFIX}:result"
    
    # MCP Server Configurations
    transformer_config: TransformerServerConfig = field(default_factory=TransformerServerConfig)
    websearch_config: WebSearchServerConfig = field(default_factory=WebSearchServerConfig)
    trajectory_config: TrajectoryServerConfig = field(default_factory=TrajectoryServerConfig)
    clustering_config: ClusteringServerConfig = field(default_factory=ClusteringServerConfig)
    
    def __init__(self, task_id: Optional[str] = None) -> None:
        self.task_id = task_id
        self.time_path = time_dir()
        self.task_path: Optional[str] = None
        
        # Initialize MCP server configs
        self.transformer_config = TransformerServerConfig()
        self.websearch_config = WebSearchServerConfig()
        self.trajectory_config = TrajectoryServerConfig()
        self.clustering_config = ClusteringServerConfig()
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.LOG_DIR,
            self.TASK_DIR,
            self.TOOL_DOC_DIR,
            self.TOOL_CODE_DIR,
            self.DATA_DIR,
            self.OUTPUT_DIR,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_task(self) -> str:
        """Get the current task ID."""
        if self.task_id is None:
            raise ValueError("Task ID is not set")
        return self.task_id
    
    def get_task_path(self) -> str:
        """Get the full path for the current task."""
        if self.task_path:
            return self.task_path
        return os.path.join(self.TASK_DIR, self.time_path, self.get_task())
    
    def get_redis_connect(self) -> redis.Redis:
        """Get a Redis connection instance."""
        return redis.Redis(
            host=self.REDIS_HOST,
            port=self.REDIS_PORT,
            db=self.REDIS_DB,
            password=self.REDIS_PASSWORD,
            decode_responses=True
        )
    
    @classmethod
    def set_task(cls, task_id: str) -> "Config":
        """Create a new Config instance with the specified task ID."""
        return Config(task_id)
    
    def log_error(self, component: str, message: str) -> None:
        """Log an error message to the error log file."""
        timestamp = datetime.now().isoformat()
        log_file = os.path.join(self.LOG_DIR, "error.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf8") as f:
            f.write(f"[{timestamp}] [{component}]\n{message}\n{'=' * 50}\n")
    
    def log_info(self, component: str, message: str) -> None:
        """Log an info message to the info log file."""
        if not self.ECHO_INFO:
            return
        timestamp = datetime.now().isoformat()
        log_file = os.path.join(self.LOG_DIR, "info.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf8") as f:
            f.write(f"[{timestamp}] [{component}] {message}\n")
        print(f"[{component}] {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "task_id": self.task_id,
            "time_path": self.time_path,
            "base_llm_model": self.BASE_LLM_MODEL,
            "super_llm_model": self.SUPER_LLM_MODEL,
            "redis_host": self.REDIS_HOST,
            "redis_port": self.REDIS_PORT,
        }


# Global config instance
config = Config()


import os
import re
import ast
import json
import hashlib
from datetime import datetime
from uuid import uuid4
from typing import Optional, Dict, Any, List, Tuple, Callable
from functools import wraps

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


# Colored print functions
def rprint(message: str) -> None:
    """Print message in red."""
    print(Fore.RED + str(message) + Style.RESET_ALL)


def gprint(message: str) -> None:
    """Print message in green."""
    print(Fore.GREEN + str(message) + Style.RESET_ALL)


def yprint(message: str) -> None:
    """Print message in yellow."""
    print(Fore.YELLOW + str(message) + Style.RESET_ALL)


def bprint(message: str) -> None:
    """Print message in blue."""
    print(Fore.BLUE + str(message) + Style.RESET_ALL)


def cprint(message: str) -> None:
    """Print message in cyan."""
    print(Fore.CYAN + str(message) + Style.RESET_ALL)


def time_dir() -> str:
    """Generate a time-based directory path."""
    now = datetime.now()
    return os.path.join(f"{now.year}-{now.month}", f"{now.day}")


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return str(uuid4())


def generate_hash(data: Any) -> str:
    """Generate a hash for given data."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    elif not isinstance(data, str):
        data = str(data)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def get_action_code(programmer_code: str, tester_code: str) -> str:
    """Combine programmer code and tester code into a single code block."""
    return f"""
{programmer_code}

{tester_code}
"""


def extract_function(code_str: str, function_name: str) -> Optional[str]:
    """Extract a function definition from code string by function name."""
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return ast.unparse(node)
    except SyntaxError:
        pass
    return None


def extract_class(code_str: str, class_name: str) -> Optional[str]:
    """Extract a class definition from code string by class name."""
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return ast.unparse(node)
    except SyntaxError:
        pass
    return None


def validate_json(json_str: str) -> Tuple[bool, Optional[Dict]]:
    """Validate and parse a JSON string."""
    try:
        data = json.loads(json_str)
        return True, data
    except json.JSONDecodeError:
        return False, None


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string with a default value on failure."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to a maximum length."""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries into one."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def deep_merge_dicts(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing invalid characters."""
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)


def ensure_dir(path: str) -> str:
    """Ensure a directory exists, create if not."""
    os.makedirs(path, exist_ok=True)
    return path


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args) -> None:
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        gprint(f"{self.name} completed in {format_duration(duration)}")
    
    @property
    def elapsed(self) -> float:
        import time
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


class RetryHandler:
    """Decorator class for retrying failed operations."""
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            retries = 0
            current_delay = self.delay
            
            while retries < self.max_retries:
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    retries += 1
                    if retries >= self.max_retries:
                        raise e
                    yprint(f"Retry {retries}/{self.max_retries} for {func.__name__}: {e}")
                    time.sleep(current_delay)
                    current_delay *= self.backoff
            
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            retries = 0
            current_delay = self.delay
            
            while retries < self.max_retries:
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    retries += 1
                    if retries >= self.max_retries:
                        raise e
                    yprint(f"Retry {retries}/{self.max_retries} for {func.__name__}: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= self.backoff
            
            return await func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper


class DataValidator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_ehr_data(ehr_list: List[str]) -> Tuple[bool, str]:
        """Validate EHR data format."""
        if not isinstance(ehr_list, list):
            return False, "EHR data must be a list"
        if len(ehr_list) == 0:
            return False, "EHR data list cannot be empty"
        for i, item in enumerate(ehr_list):
            if not isinstance(item, str):
                return False, f"EHR item at index {i} must be a string"
        return True, "Valid"
    
    @staticmethod
    def validate_lab_tests(lab_tests: List[List[Any]]) -> Tuple[bool, str]:
        """Validate lab test data format (2D array)."""
        if not isinstance(lab_tests, list):
            return False, "Lab tests must be a list"
        if len(lab_tests) == 0:
            return False, "Lab tests list cannot be empty"
        for i, row in enumerate(lab_tests):
            if not isinstance(row, list):
                return False, f"Lab test row at index {i} must be a list"
            if len(row) == 0:
                return False, f"Lab test row at index {i} cannot be empty"
        return True, "Valid"
    
    @staticmethod
    def validate_input_data(data: Dict) -> Tuple[bool, str]:
        """Validate complete input data structure."""
        if "ehr" not in data:
            return False, "Missing 'ehr' field in input data"
        if "lab_tests" not in data:
            return False, "Missing 'lab_tests' field in input data"
        
        valid, msg = DataValidator.validate_ehr_data(data["ehr"])
        if not valid:
            return False, msg
        
        valid, msg = DataValidator.validate_lab_tests(data["lab_tests"])
        if not valid:
            return False, msg
        
        return True, "Valid"


class XMLParser:
    """Utility class for parsing XML-tagged content from LLM responses."""
    
    @staticmethod
    def get_tag_content(text: str, tag_name: str) -> Optional[str]:
        """Extract content from a single XML tag."""
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        results = re.findall(pattern, text, re.DOTALL)
        return results[0] if results else None
    
    @staticmethod
    def get_tag_list_content(text: str, tag_name: str) -> List[str]:
        """Extract content from all matching XML tags."""
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        return re.findall(pattern, text, re.DOTALL)
    
    @staticmethod
    def has_tag(text: str, tag_name: str) -> bool:
        """Check if text contains a specific XML tag."""
        pattern = f'<{tag_name}>.*?</{tag_name}>'
        return bool(re.search(pattern, text, re.DOTALL))
    
    @staticmethod
    def count_tags(text: str, tag_name: str) -> int:
        """Count occurrences of a specific XML tag."""
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        return len(re.findall(pattern, text, re.DOTALL))
    
    @staticmethod
    def get_multiple_tags(text: str, *tag_names: str) -> Dict[str, Optional[str]]:
        """Extract content from multiple XML tags."""
        return {tag: XMLParser.get_tag_content(text, tag) for tag in tag_names}


def knowledge_base_store(
    item_key: str,
    query_key: str,
    query_value: str,
    question_id: str,
    task_id: Optional[str] = None
) -> None:
    """Store knowledge in the memory knowledge base."""
    from config import Config
    
    r = Config().get_redis_connect()
    memory_id = generate_task_id()
    
    data = {
        "key": query_key,
        "value": query_value,
        "id": memory_id,
        "question_id": question_id,
        "task_id": task_id,
        "item_key": item_key,
        "timestamp": datetime.now().isoformat()
    }
    
    r.lpush(f"{Config.REDIS_MEMORY_STORAGE_KEY}:{item_key}", json.dumps(data))
    r.hset(Config.REDIS_MEMORY_INFO_KEY, memory_id, json.dumps(data))


def knowledge_base_retrieve(
    item_key: str,
    query: str,
    k: int = 5
) -> List[Dict]:
    """Retrieve relevant knowledge from the memory knowledge base."""
    from config import Config
    
    r = Config().get_redis_connect()
    key = f"{Config.REDIS_MEMORY_STORAGE_KEY}:{item_key}"
    
    # Get all stored items
    items = r.lrange(key, 0, -1)
    results = []
    
    for item in items:
        data = safe_json_loads(item, {})
        if data:
            results.append(data)
    
    # Return top k results (in a real system, this would use semantic search)
    return results[:k]


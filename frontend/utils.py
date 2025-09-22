import requests
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def format_datetime(datetime_str: str) -> str:
    """Format datetime string for display"""
    try:
        if 'T' in datetime_str:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(datetime_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return datetime_str[:16]  # Fallback to first 16 characters

def calculate_elapsed_time(start_time_str: str) -> str:
    """Calculate elapsed time from start time string"""
    try:
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        elapsed = datetime.now() - start_time.replace(tzinfo=None)
        
        total_seconds = int(elapsed.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except:
        return "Unknown"

def check_backend_health(backend_url: str) -> Tuple[bool, Optional[Dict]]:
    """Check if backend is healthy and responsive"""
    try:
        response = requests.get(f"{backend_url}/", timeout=3)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except:
        return False, None

def export_chat_history(chat_history: List) -> Optional[str]:
    """Export chat history as JSON"""
    if chat_history:
        chat_data = {
            "exported_at": datetime.now().isoformat(),
            "total_messages": len(chat_history),
            "conversations": []
        }
        
        for question, answer, sources, timestamp in chat_history:
            chat_data["conversations"].append({
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": timestamp
            })
        
        return json.dumps(chat_data, indent=2)
    return None

def initialize_session_state() -> Dict[str, Any]:
    """Initialize all session state variables"""
    import streamlit as st
    
    defaults = {
        'chat_history': [],
        'last_status_check': 0,
        'active_tasks': {},
        'streaming_response': None,
        'confirm_clear': False,
        'backend_connected': False,
        'db_status': {},
        'documents_data': {},  # Changed from None to {}
        'backend_info': {}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def cleanup_old_tasks(active_tasks: Dict[str, Dict]) -> None:
    """Clean up old completed/failed tasks"""
    current_time = time.time()
    tasks_to_remove = []
    
    for task_id, task_info in active_tasks.items():
        if task_info['status'] in ['completed', 'failed']:
            completion_time = task_info.get('completion_time', current_time)
            if current_time - completion_time > 60:  # Remove after 1 minute
                tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del active_tasks[task_id]

def update_task_statuses(backend_url: str, active_tasks: Dict[str, Dict]) -> None:
    """Update status of active tasks"""
    tasks_to_remove = []
    
    for task_id, task_info in active_tasks.items():
        if task_info['status'] == 'processing':
            try:
                response = requests.get(f"{backend_url}/task/{task_id}", timeout=5)
                if response.status_code == 200:
                    task_status = response.json()
                    active_tasks[task_id].update({
                        'status': task_status['status'],
                        'progress': task_status.get('progress', 0),
                        'message': task_status.get('message', ''),
                        'error': task_status.get('error')
                    })
                    
                    # Mark completion time for completed/failed tasks
                    if task_status['status'] in ['completed', 'failed']:
                        if not task_info.get('completion_time'):
                            active_tasks[task_id]['completion_time'] = time.time()
                        elif time.time() - task_info['completion_time'] > 30:
                            tasks_to_remove.append(task_id)
            except Exception as e:
                logging.error(f"Error updating task {task_id}: {e}")
    
    # Remove old tasks
    for task_id in tasks_to_remove:
        del active_tasks[task_id]

def validate_file_upload(file, supported_formats: List[str]) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if not file:
        return False, "No file selected"
    
    import os
    file_ext = os.path.splitext(file.name)[1].lower()
    
    if file_ext not in supported_formats:
        return False, f"Unsupported format. Supported: {', '.join(supported_formats)}"
    
    # Check file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    if file.size > max_size:
        return False, f"File too large. Max size: {format_file_size(max_size)}"
    
    return True, "Valid file"

def make_api_request(url: str, method: str = "GET", **kwargs) -> Tuple[bool, Any]:
    """Make API request with error handling"""
    try:
        if method.upper() == "GET":
            response = requests.get(url, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, **kwargs)
        elif method.upper() == "DELETE":
            response = requests.delete(url, **kwargs)
        else:
            return False, f"Unsupported method: {method}"
        
        if response.status_code == 200:
            try:
                return True, response.json()
            except:
                return True, response.text
        else:
            try:
                error_data = response.json()
                return False, error_data.get('detail', 'Unknown error')
            except:
                return False, f"HTTP {response.status_code}: {response.text}"
                
    except requests.exceptions.Timeout:
        return False, "Request timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, str(e)

def parse_streaming_response(response_text: str) -> List[Dict]:
    """Parse server-sent events streaming response"""
    events = []
    lines = response_text.strip().split('\n')
    
    current_event = {}
    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            try:
                data = json.loads(line[6:])  # Remove 'data: ' prefix
                events.append(data)
            except json.JSONDecodeError:
                continue
        elif line == '':
            # Empty line indicates end of event
            if current_event:
                events.append(current_event)
                current_event = {}
    
    return events

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    import platform
    import sys
    
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat()
    }

def log_user_action(action: str, details: Dict[str, Any] = None) -> None:
    """Log user actions for debugging"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details or {}
    }
    logging.info(f"User action: {json.dumps(log_entry)}")

def create_progress_bar_text(progress: int, width: int = 20) -> str:
    """Create a text-based progress bar"""
    filled = int(progress * width / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {progress}%"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    # Trim underscores from start and end
    filename = filename.strip('_')
    return filename

def calculate_processing_speed(start_time: str, items_processed: int) -> str:
    """Calculate processing speed"""
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        elapsed = datetime.now() - start.replace(tzinfo=None)
        elapsed_seconds = elapsed.total_seconds()
        
        if elapsed_seconds > 0 and items_processed > 0:
            speed = items_processed / elapsed_seconds
            if speed >= 1:
                return f"{speed:.1f} items/sec"
            else:
                return f"{60/speed:.1f} sec/item"
        return "Calculating..."
    except:
        return "Unknown"

def format_error_message(error: Any) -> str:
    """Format error message for display"""
    if isinstance(error, dict):
        return error.get('detail', str(error))
    elif isinstance(error, Exception):
        return str(error)
    else:
        return str(error)

def is_valid_url(url: str) -> bool:
    """Check if URL is valid"""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def create_backup_filename(original_name: str) -> str:
    """Create backup filename with timestamp"""
    import os
    name, ext = os.path.splitext(original_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_backup_{timestamp}{ext}"

def get_file_info(filepath: str) -> Dict[str, Any]:
    """Get detailed file information"""
    import os
    try:
        stat = os.stat(filepath)
        return {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": os.path.splitext(filepath)[1].lower(),
            "exists": True
        }
    except:
        return {"exists": False}

class TaskTracker:
    """Helper class for tracking task progress"""
    
    def __init__(self):
        self.tasks = {}
    
    def add_task(self, task_id: str, task_type: str, description: str = ""):
        """Add a new task"""
        self.tasks[task_id] = {
            'type': task_type,
            'status': 'processing',
            'progress': 0,
            'message': description,
            'start_time': time.time(),
            'completion_time': None
        }
    
    def update_task(self, task_id: str, progress: int = None, message: str = None, status: str = None):
        """Update task progress"""
        if task_id in self.tasks:
            if progress is not None:
                self.tasks[task_id]['progress'] = progress
            if message is not None:
                self.tasks[task_id]['message'] = message
            if status is not None:
                self.tasks[task_id]['status'] = status
                if status in ['completed', 'failed']:
                    self.tasks[task_id]['completion_time'] = time.time()
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task information"""
        return self.tasks.get(task_id)
    
    def remove_task(self, task_id: str):
        """Remove task from tracking"""
        if task_id in self.tasks:
            del self.tasks[task_id]
    
    def cleanup_old_tasks(self, max_age: int = 300):
        """Remove tasks older than max_age seconds"""
        current_time = time.time()
        tasks_to_remove = []
        
        for task_id, task_info in self.tasks.items():
            if task_info['status'] in ['completed', 'failed']:
                completion_time = task_info.get('completion_time', current_time)
                if current_time - completion_time > max_age:
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            self.remove_task(task_id)
    
    def get_active_tasks(self) -> List[Dict]:
        """Get all active (processing) tasks"""
        return [
            {**task_info, 'id': task_id} 
            for task_id, task_info in self.tasks.items() 
            if task_info['status'] == 'processing'
        ]

# Global task tracker instance
task_tracker = TaskTracker()

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

def get_memory_usage() -> Dict[str, str]:
    """Get current memory usage"""
    import psutil
    import os
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss": format_file_size(memory_info.rss),
            "vms": format_file_size(memory_info.vms),
            "percent": f"{process.memory_percent():.1f}%"
        }
    except:
        return {"error": "Unable to get memory info"}
import psutil
import os

def get_system_snapshot():
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        "ram_available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "ram_percent": psutil.virtual_memory().percent
    }


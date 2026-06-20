import psutil
import os

def get_system_snapshot():
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        "ram_available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "ram_percent": psutil.virtual_memory().percent
    }

def get_resource_delta(before: dict, after: dict):
    return {
        "cpu_peak_percent": after["cpu_percent"],
        "ram_delta_mb": after["ram_used_mb"] - before["ram_used_mb"],
        "ram_peak_percent": after["ram_percent"]
    }
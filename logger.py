import os
import hashlib
import json

LOG_FILE = "file_log.json"

def _compute_file_hash(path):
    """Compute SHA256 hash of a file's contents."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None

def _load_log():
    """Load log file or return an empty dictionary."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_log(log_data):
    """Save log data to a file."""
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)

def is_logged(path) -> bool:
    """Check if the file is logged and its content is unchanged."""
    log_data = _load_log()
    if path not in log_data:
        return False

    current_hash = _compute_file_hash(path)
    return log_data[path]["hash"] == current_hash

def log_file(path) -> None:
    """Log the file with its current hash."""
    log_data = _load_log()
    file_hash = _compute_file_hash(path)
    
    if file_hash is not None:
        log_data[path] = {"hash": file_hash}
        _save_log(log_data)

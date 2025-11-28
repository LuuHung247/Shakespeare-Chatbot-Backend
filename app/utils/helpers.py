"""Helper functions."""

def ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)

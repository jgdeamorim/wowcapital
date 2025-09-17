import os
from pathlib import Path


def get_secret(name: str) -> str:
    """Resolve secret by env var NAME or file path NAME_FILE.
    Precedence: NAME_FILE (if set and file exists) -> NAME -> error.
    This enables Docker/K8s secrets mounted as files.
    """
    file_var = f"{name}_FILE"
    file_path = os.getenv(file_var)
    if file_path and Path(file_path).exists():
        try:
            return Path(file_path).read_text().strip()
        except Exception:
            pass
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing secret env var: {name}")
    return val

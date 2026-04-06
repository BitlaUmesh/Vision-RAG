"""
storage.py — Cloud Storage Abstraction
================================================================================
Handles document storage (Local Disk vs AWS S3).
"""

import os
import io
from abc import ABC, abstractmethod
from pathlib import Path

logger = importlib_logger = __import__('logging').getLogger(__name__)

class StorageProvider(ABC):
    @abstractmethod
    def save_file(self, filename: str, content: bytes) -> str:
        """Saves a file and returns its access URI/path."""
        pass

    @abstractmethod
    def get_file(self, filename: str) -> bytes:
        """Retrieves file content."""
        pass

    @abstractmethod
    def delete_file(self, filename: str) -> bool:
        """Deletes a file."""
        pass

    @abstractmethod
    def list_files(self) -> list[str]:
        """Lists all files in storage."""
        pass

    @abstractmethod
    def get_local_path(self, filename: str) -> str:
        """Returns a local file path. If cloud storage, downloads temporarily."""
        pass


class LocalStorageProvider(StorageProvider):
    def __init__(self, upload_dir: str = "data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, filename: str, content: bytes) -> str:
        path = self.upload_dir / filename
        with open(path, "wb") as f:
            f.write(content)
        return str(path)

    def get_file(self, filename: str) -> bytes:
        path = self.upload_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{filename} not found.")
        with open(path, "rb") as f:
            return f.read()

    def delete_file(self, filename: str) -> bool:
        path = self.upload_dir / filename
        if path.exists():
            path.unlink()
            return True
        return False

    def list_files(self) -> list[str]:
        return [p.name for p in self.upload_dir.glob("*.pdf")]

    def get_local_path(self, filename: str) -> str:
        return str(self.upload_dir / filename)


def get_storage() -> StorageProvider:
    return LocalStorageProvider()

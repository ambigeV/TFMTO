"""File and folder management utilities."""

import shutil
from pathlib import Path


class FileManager:
    """Manage Data and Results folders."""

    def __init__(self, base_path: str = "./tests"):
        self.base_path = Path(base_path).resolve()
        self.data_path = self.base_path / "Data"
        self.results_path = self.base_path / "Results"
        self.backup_path = self.base_path / "backup"

    def ensure_structure(self):
        """Create folder structure if not exists."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)

    def clean_data(self):
        """Remove all files in Data/."""
        if self.data_path.exists():
            shutil.rmtree(self.data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def clean_results(self):
        """Remove all files in Results/."""
        if self.results_path.exists():
            shutil.rmtree(self.results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)

    def clean_all(self):
        """Remove all files in Data/ and Results/."""
        self.clean_data()
        self.clean_results()

    def get_pkl_count(self) -> int:
        """Count .pkl files in Data/."""
        return len(list(self.data_path.rglob("*.pkl")))

    def data_exists(self) -> bool:
        """Check if any data files exist."""
        return self.get_pkl_count() > 0

    def results_exist(self) -> bool:
        """Check if any result files exist."""
        if not self.results_path.exists():
            return False
        return any(self.results_path.rglob("*"))

    def get_data_path_str(self) -> str:
        """Get data path as string."""
        return str(self.data_path)

    def get_results_path_str(self) -> str:
        """Get results path as string."""
        return str(self.results_path)

    def get_base_path_str(self) -> str:
        """Get base path as string."""
        return str(self.base_path)

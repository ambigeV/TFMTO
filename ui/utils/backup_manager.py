"""Backup and restore utilities."""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class BackupManager:
    """Handle backup and restore operations."""

    def __init__(self, base_path: str = "./tests"):
        self.base_path = Path(base_path).resolve()
        self.backup_path = self.base_path / "backup"

    def ensure_backup_folder(self):
        """Create backup folder if not exists."""
        self.backup_path.mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> Optional[str]:
        """
        Create timestamped backup of Data/ and Results/.

        Returns:
            Path to created archive, or None if nothing to backup.
        """
        self.ensure_backup_folder()

        # Check if there's anything to backup
        data_path = self.base_path / "Data"
        results_path = self.base_path / "Results"

        has_data = data_path.exists() and any(data_path.rglob("*"))
        has_results = results_path.exists() and any(results_path.rglob("*"))

        if not has_data and not has_results:
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f'data_{timestamp}.zip'
        archive_path = self.backup_path / archive_name

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for folder_name in ['Data', 'Results']:
                folder_path = self.base_path / folder_name
                if folder_path.exists():
                    for file in folder_path.rglob('*'):
                        if file.is_file():
                            arcname = file.relative_to(self.base_path)
                            zf.write(file, arcname)

        return str(archive_path)

    def list_backups(self) -> List[Dict]:
        """
        List available backups with metadata.

        Returns:
            List of dicts with 'name', 'path', 'size', 'date' keys.
        """
        self.ensure_backup_folder()
        backups = []

        for f in sorted(self.backup_path.glob("data_*.zip"), reverse=True):
            stat = f.stat()
            # Parse timestamp from filename
            try:
                name = f.stem  # data_YYYYMMDD_HHMMSS
                ts_str = name.replace("data_", "")
                dt = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
                date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                date_str = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

            backups.append({
                'name': f.name,
                'path': str(f),
                'size': stat.st_size,
                'size_str': self._format_size(stat.st_size),
                'date': date_str,
            })

        return backups

    def restore_backup(self, archive_path: str) -> bool:
        """
        Restore from backup archive.

        Args:
            archive_path: Path to the zip archive.

        Returns:
            True if successful, False otherwise.
        """
        archive = Path(archive_path)
        if not archive.exists():
            return False

        try:
            with zipfile.ZipFile(archive, 'r') as zf:
                zf.extractall(self.base_path)
            return True
        except Exception:
            return False

    def delete_backup(self, archive_path: str) -> bool:
        """
        Delete a backup archive.

        Args:
            archive_path: Path to the zip archive.

        Returns:
            True if successful, False otherwise.
        """
        archive = Path(archive_path)
        if archive.exists() and archive.suffix == '.zip':
            try:
                archive.unlink()
                return True
            except Exception:
                return False
        return False

    def clean_and_backup(self) -> Optional[str]:
        """
        Create backup, then clean Data/ and Results/.

        Returns:
            Path to backup if created, None otherwise.
        """
        backup_path = self.create_backup()

        # Clean Data/ and Results/
        for folder_name in ['Data', 'Results']:
            folder_path = self.base_path / folder_name
            if folder_path.exists():
                shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)

        return backup_path

    def clean_without_backup(self):
        """Clean Data/ and Results/ without backup."""
        for folder_name in ['Data', 'Results']:
            folder_path = self.base_path / folder_name
            if folder_path.exists():
                shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

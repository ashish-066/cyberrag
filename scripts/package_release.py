import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DESKTOP_DIR = Path(r"C:\Users\njosh\Desktop")

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    ".gemini",
    "scratch",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".zip"}

def should_include(rel_path: Path) -> bool:
    parts = rel_path.parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False
    if rel_path.suffix.lower() in EXCLUDE_EXTS:
        return False
    return True

def package_release(target_zip: Path):
    print(f"Packaging {target_zip.name}...")
    files_added = 0
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ROOT_DIR):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                abs_file = Path(root) / file
                rel_file = abs_file.relative_to(ROOT_DIR)
                if should_include(rel_file):
                    # Use forward slashes for cross-platform zip format
                    arcname = rel_file.as_posix()
                    zf.write(abs_file, arcname)
                    files_added += 1

    print(f"Successfully packaged {files_added} files into {target_zip} ({target_zip.stat().st_size / (1024*1024):.2f} MB)")
    return files_added

if __name__ == "__main__":
    targets = [
        DESKTOP_DIR / "CyberRAG_Final_Release_v7_polished_ui.zip",
        DESKTOP_DIR / "CyberRAG_Final_Release_v6_polished_demo.zip",
        DESKTOP_DIR / "CyberRAG_Final_Release_v5.zip",
        DESKTOP_DIR / "CyberRAG_Final_Release_v4.zip",
        DESKTOP_DIR / "CyberRAG_Final_Release.zip",
    ]
    for target in targets:
        count = package_release(target)

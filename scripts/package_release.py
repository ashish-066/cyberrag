import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("CYBERRAG_DIST_DIR", ROOT_DIR / "dist"))

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    ".gemini",
    "dist",
    ".hypothesis",
    "scratch",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd"}

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
    import sys

    version = sys.argv[1] if len(sys.argv) > 1 else "v6"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    package_release(OUT_DIR / f"CyberRAG_Final_Release_{version}.zip")

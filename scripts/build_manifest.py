"""Write data/MANIFEST.json (sha256 of every required data file). Run after intentional data changes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.bootstrap import build_manifest, manifest_hash  # noqa: E402
from src.core.config import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    manifest = build_manifest(settings)
    settings.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {settings.manifest_path} ({len(manifest['files'])} files, hash {manifest_hash(manifest)[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

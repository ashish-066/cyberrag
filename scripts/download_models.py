"""One-time model download (run while online). After this, everything works offline.

Fetches only the PyTorch weights and tokenizer files, not the ONNX / OpenVINO
variants that the model repos also contain.
"""
from __future__ import annotations

import os
import sys

MODELS = {
    "cross-encoder/nli-deberta-v3-base": None,
    "sentence-transformers/all-MiniLM-L6-v2": None,
}
ALLOW = [
    "*.json", "*.txt", "*.model", "spm.model", "*.safetensors", "pytorch_model.bin",
    "modules.json", "sentence_bert_config.json", "1_Pooling/*", "tokenizer*", "vocab*",
]
IGNORE = ["onnx/*", "openvino/*", "*.onnx", "*.msgpack", "*.h5", "*.ot", "tf_model*", "*.csv", "README.md"]


def main() -> int:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed; run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    for repo in MODELS:
        print(f"Downloading {repo} ...")
        try:
            path = snapshot_download(repo, allow_patterns=ALLOW, ignore_patterns=IGNORE)
        except Exception as exc:
            print(f"FAILED {repo}: {exc}", file=sys.stderr)
            return 1
        print(f"  ok -> {path}")
    print("Done. Set CTI_OFFLINE=1 (the default) to run without network access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

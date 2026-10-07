"""Explicit, repository-local setup for the optional original LaMini weights."""
from pathlib import Path
import os


def main():
    root = Path(__file__).resolve().parent
    os.environ.setdefault("HF_HOME", str(root / "models" / "hf-cache"))
    from huggingface_hub import snapshot_download
    destination = snapshot_download(
        "MBZUAI/LaMini-Flan-T5-248M",
        local_dir=root / "Lamini-1",
        allow_patterns=["*.json", "*.model", "*.safetensors", "pytorch_model.bin"],
    )
    print(f"Model is ready at {destination}")


if __name__ == "__main__":
    main()

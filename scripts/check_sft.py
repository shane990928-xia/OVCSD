"""Check the two original ALFWorld SFT weight shards before an exact-start run."""
import argparse
import hashlib
from pathlib import Path

SHARDS = {
    "model-00001-of-00002.safetensors": "0aa398a38c7370c646af7660bf287493e299a2d9a9e28dabd242cc9f1d705f5e",
    "model-00002-of-00002.safetensors": "2f0c3d555eeea3211eef03d090fecd2d40f00ed277bb8b8868d6f98a464447e9",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    checkpoint = parser.parse_args().checkpoint
    for filename, expected in SHARDS.items():
        with (checkpoint / filename).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise SystemExit(f"{filename} has SHA-256 {actual}, expected {expected}")
        print(f"{actual}  {filename}")


if __name__ == "__main__":
    main()

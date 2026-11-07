"""Generate the eight training slots and fixed seen validation set."""
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def row(split, index):
    return {"data_source": "text", "prompt": [{"role": "user", "content": ""}],
            "ability": "agent", "extra_info": {"split": split, "index": index}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    games = args.assets.resolve() / "json_2.1.1"
    if not (games / "train").is_dir():
        raise FileNotFoundError(games / "train")
    manifest = Path(__file__).resolve().parents[1] / "memory_data/alfworld/valid_seen_128.txt"
    validation = []
    for i, rel in enumerate(manifest.read_text().splitlines()):
        game = games / "valid_seen" / rel
        if not game.is_file():
            raise FileNotFoundError(game)
        record = row("test", i)
        record["extra_info"]["task_id"] = "extra.gamefile:" + str(game)
        record["env_kwargs"] = {"gamefile": str(game)}
        validation.append(record)
    if len(validation) != 128:
        raise ValueError("Expected 128 held-out seen games")
    args.output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row("train", i) for i in range(8)]), args.output / "train.parquet")
    pq.write_table(pa.Table.from_pylist(validation), args.output / "val.parquet")


if __name__ == "__main__":
    main()

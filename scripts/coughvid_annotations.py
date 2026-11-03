"""Flattens COUGHVID per-recording JSON annotations into CSV tables.

Writes:
    <table>: one row per recording, nested expert labels flattened to
        `expert_labels_N.<field>` columns, plus the audio file name and format.
    <out-dir>/field_stats.csv: how many recordings carry each field.
    <out-dir>/field_values.csv: value counts for enum-like fields.

Usage:
    .venv/bin/python scripts/coughvid_annotations.py
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

AUDIO_SUFFIXES = (".webm", ".ogg", ".wav")
# Fields with more distinct values are free-form (age, coordinates, timestamps) and
# would flood field_values.csv, so only their presence is reported.
MAX_ENUM_VALUES = 20


def flatten(annotation: dict, prefix: str = "") -> dict[str, str]:
    """Flattens nested dicts into dotted keys.

    Args:
        annotation: Parsed JSON object; values may be nested dicts.
        prefix: Key prefix for the current nesting level.

    Returns:
        Mapping from dotted field path to its string value.
    """
    flat = {}
    for key, value in annotation.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = str(value)
    return flat


def load_records(dataset_dir: Path) -> list[dict[str, str]]:
    """Reads every `<uuid>.json` and pairs it with its audio file.

    Args:
        dataset_dir: Folder with `<uuid>.json` and `<uuid>.<audio>` files.

    Returns:
        One flat record per JSON, with `uuid`, `audio_file` and `audio_format` added;
        both audio fields are empty when the recording has no audio file.
    """
    audio_by_uuid = {
        path.stem: path.name
        for path in dataset_dir.iterdir()
        if path.suffix in AUDIO_SUFFIXES
    }
    records = []
    for json_path in sorted(dataset_dir.glob("*.json")):
        audio_file = audio_by_uuid.get(json_path.stem, "")
        records.append(
            {
                "uuid": json_path.stem,
                "audio_file": audio_file,
                "audio_format": Path(audio_file).suffix.lstrip("."),
                **flatten(json.loads(json_path.read_text())),
            }
        )
    return records


def write_csv(path: Path, header: list[str], rows: list[list | dict]) -> None:
    """Writes rows (lists, or dicts keyed by header) to a CSV file, creating parents.

    Args:
        path: Output file.
        header: Column names.
        rows: Lists in header order, or dicts; missing dict keys become empty cells.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        if rows and isinstance(rows[0], dict):
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
        else:
            writer = csv.writer(file)
            writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/coughvid/coughvid_20211012")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/coughvid"))
    # Dataset-sized derived data, so it sits next to the dataset (git-ignored).
    parser.add_argument(
        "--table", type=Path, default=Path("data/coughvid/annotations.csv")
    )
    args = parser.parse_args()

    records = load_records(args.dataset_dir)
    # Fields in first-seen order, so the table keeps the source JSON layout.
    fields = list(dict.fromkeys(field for record in records for field in record))
    values_by_field = {
        field: Counter(record[field] for record in records if field in record)
        for field in fields
    }

    write_csv(args.table, fields, records)

    stats_rows = []
    values_rows = []
    for field, values in values_by_field.items():
        present = values.total()
        is_enum = len(values) <= MAX_ENUM_VALUES
        stats_rows.append(
            [
                field,
                present,
                round(100 * present / len(records), 2),
                len(values),
                is_enum,
            ]
        )
        if is_enum:
            values_rows += [
                [field, value, count, round(100 * count / present, 2)]
                for value, count in values.most_common()
            ]
    write_csv(
        args.out_dir / "field_stats.csv",
        ["field", "present", "present_pct", "n_unique", "is_enum"],
        stats_rows,
    )
    write_csv(
        args.out_dir / "field_values.csv",
        ["field", "value", "count", "pct_of_present"],
        values_rows,
    )
    print(f"{len(records)} recordings, {len(fields)} fields")


if __name__ == "__main__":
    main()

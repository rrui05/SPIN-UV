#!/usr/bin/env python
"""Materialize the E2 shared-label taxonomy as CSV and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path


E2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = E2_ROOT.parent


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    taxonomy = read_json(E2_ROOT / "configs" / "common_taxonomy_shared_selected_labels.json")
    classes_payload = read_json(EXPERIMENT_ROOT / "data" / "processed" / "spinuv_semantic" / "meta" / "classes.json")
    train_id_by_name = {item["name"]: int(item["train_id"]) for item in classes_payload["classes"]}
    rows = []
    for name in taxonomy["main_comparison_classes"]:
        rows.append(
            {
                "class_name": name,
                "spinuv_train_id": train_id_by_name[name],
                "included_in_main_comparison": 1,
                "reason": "semantically_consistent_shared_label",
            }
        )
    for item in taxonomy["excluded_from_main_comparison"]:
        name = item["spinuv_label"]
        rows.append(
            {
                "class_name": name,
                "spinuv_train_id": train_id_by_name[name],
                "included_in_main_comparison": 0,
                "reason": item["reason"],
            }
        )

    tables_dir = E2_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / "experiment_02_common_taxonomy.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (tables_dir / "experiment_02_common_taxonomy.json").open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)
        f.write("\n")
    print(json.dumps({"rows": len(rows), "included": len(taxonomy["main_comparison_classes"])}, indent=2))


if __name__ == "__main__":
    main()

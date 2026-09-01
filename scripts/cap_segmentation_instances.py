"""Create an auditable bounded reconstruction subset from a full segmentation response."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--max-instances", type=int, default=21)
    args = parser.parse_args()
    response = args.run.resolve() / "work" / "segmentation_response.json"
    value = json.loads(response.read_text(encoding="utf-8"))
    instances = list(value.get("instances", []))
    if len(instances) <= args.max_instances:
        print(json.dumps({"status": "unchanged", "instances": len(instances)}))
        return 0
    backup = response.with_name("segmentation_response.full_instances.json")
    if not backup.exists():
        shutil.copy2(response, backup)
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in instances:
        groups[item.get("category", "unknown")].append(item)
    categories = sorted(groups)
    quota = max(1, args.max_instances // len(categories))
    selected: list[dict] = []
    for category in categories:
        selected.extend(sorted(groups[category], key=lambda x: float(x.get("score", 0)), reverse=True)[:quota])
    selected_ids = {id(item) for item in selected}
    remaining = sorted(
        (item for item in instances if id(item) not in selected_ids),
        key=lambda x: float(x.get("score", 0)), reverse=True,
    )
    selected.extend(remaining[: max(0, args.max_instances - len(selected))])
    selected.sort(key=lambda x: (x.get("category", ""), -float(x.get("score", 0))))
    value["instances"] = selected
    value["contract_repair"] = {
        **value.get("contract_repair", {}),
        "kind": "bounded_reconstruction_subset",
        "max_instances": args.max_instances,
        "full_instances": len(instances),
        "selected_instances": len(selected),
        "source_backup": str(backup),
    }
    response.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "capped", "full_instances": len(instances),
                      "selected_instances": len(selected), "backup": str(backup)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

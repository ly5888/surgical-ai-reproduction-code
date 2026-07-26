#!/usr/bin/env python3
"""Non-anchored v2 entry point for the Sum-of-Checks reconstruction.

Place this file beside sumofchecks_eval.py. It reuses the validated v1 data,
API, parsing, resume and metrics code, while replacing only the negatively
anchored output-format prompts. It also creates a deterministic 8-image
prompt-development manifest disjoint from the formal 30-image evaluation set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import sumofchecks_eval as base
except ImportError as exc:
    raise SystemExit(
        "Put sumofchecks_eval_v2.py in the same scripts directory as "
        "sumofchecks_eval.py"
    ) from exc


PROMPT_VERSION = "nonanchored-v2.0"
DEV_SEED = 20260720

DIRECT_PROMPT_V2 = """Assess the Critical View of Safety (CVS) in this single laparoscopic cholecystectomy frame.

C1: Exactly two and only two tubular structures visibly enter the gallbladder.
C2: The hepatocystic triangle is cleared of fat and fibrous tissue.
C3: The lower third of the gallbladder is detached from the liver bed.

Assess C1, C2, and C3 independently from visible evidence in the image.
Assign integer 1 only when the criterion is clearly demonstrated.
Assign integer 0 when it is absent, incomplete, obscured, or uncertain.
Neither value is a default. Do not infer one criterion from another.

Return one valid JSON object only, without markdown or commentary.
The object must contain exactly the keys C1, C2, and C3.
The value of each key must be the integer 0 or 1.
"""


def soc_prompt_v2(criterion: str) -> str:
    spec = base.CRITERIA[criterion]
    checks = "\n".join(f"{rid}: {text}" for rid, text, _ in spec["checks"])
    keys = ", ".join(rid for rid, _, _ in spec["checks"])
    return f"""Evaluate {criterion} in this single laparoscopic cholecystectomy frame.

{criterion} is satisfied only when: {spec['definition']}

Evaluate these checks independently from visible evidence:
{checks}

For every check, choose exactly one judgment:
- yes: clear visible evidence supports the check;
- no: visible evidence contradicts the check or shows it is absent;
- uncertain: the image does not provide enough visibility to decide.

No judgment is the default. Do not copy a judgment from these instructions.
Do not infer an anatomical finding solely from the expected surgical workflow.

Return one valid JSON object only, without markdown or commentary.
The top-level object must contain exactly these keys: {keys}.
Each key's value must be an object containing exactly two fields:
- judgment: one of the strings yes, no, or uncertain;
- reason: one short sentence describing the visible evidence.
"""


def prompt_hashes() -> dict[str, str]:
    prompts = {"direct": DIRECT_PROMPT_V2}
    prompts.update({f"soc_{c.lower()}": soc_prompt_v2(c) for c in base.CRITERIA})
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in prompts.items()
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_identity(row: dict[str, str]) -> str:
    return row.get("image_path") or row.get("file_name") or row.get("image_id", "")


def make_dev_manifest(args: argparse.Namespace) -> None:
    source = read_csv(args.source)
    formal = read_csv(args.formal_manifest)
    replacement = read_csv(args.replacement_manifest) if args.replacement_manifest.exists() else []
    excluded = {row_identity(r) for r in formal + replacement}
    excluded_names = {"173_23100.jpg"}

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source:
        identity = row_identity(row)
        name = row.get("file_name") or Path(row.get("image_path", "")).name
        if identity in excluded or name in excluded_names:
            continue
        groups[base.label(row)].append(row)

    expected = [f"{n:03b}" for n in range(8)]
    missing = [key for key in expected if not groups.get(key)]
    if missing:
        raise ValueError(f"Source has no eligible rows for label combinations: {missing}")

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for dev_index, key in enumerate(expected):
        population = sorted(
            groups[key],
            key=lambda r: (int(r.get("sample_index", 10**9)), row_identity(r)),
        )
        row = dict(rng.choice(population))
        row["stratified_index"] = str(1000 + dev_index)
        row["dev_index"] = str(dev_index)
        row["label_combo"] = key
        selected.append(row)

    # Match the formal manifest's schema so the base runner can consume it.
    formal_fields = list(formal[0])
    fieldnames = ["stratified_index", "dev_index", "label_combo"] + [
        name for name in formal_fields
        if name not in {"stratified_index", "dev_index", "label_combo"}
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Development manifest: {args.output}")
    print(f"Rows: {len(selected)}  Seed: {args.seed}  SHA256: {digest}")
    for row in selected:
        print(row["dev_index"], row["label_combo"], row.get("file_name") or Path(row["image_path"]).name)


def write_audit_config(args: argparse.Namespace) -> None:
    config_path = Path(str(args.output) + ".config.json")
    config = {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hashes(),
        "method": args.method,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "threshold": 0.5 if args.method == "soc-nofs" else None,
        "notes": [
            "No filled all-zero Direct JSON example is present.",
            "No filled all-no SoC judgment example is present.",
            "SoC checks and weights remain the fixed reconstructed v1 definitions.",
        ],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_base_run = base.run


def run_v2(args: argparse.Namespace) -> None:
    # Freeze the v2 inference settings. Output length is raised for 7-check C3 JSON.
    args.temperature = 0.0
    if args.max_tokens == 500:
        args.max_tokens = 1000
    write_audit_config(args)
    _base_run(args)


def development_report(args: argparse.Namespace) -> None:
    rows = base.load_latest(args.results)
    if not rows:
        raise ValueError("No successful rows found")
    unique = list(rows.values())
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Completed: {len(unique)}/8")
    print("criterion  predicted_positive  nonzero_score  score_min  score_mean  score_max")
    for c in base.CRITERIA:
        predictions = [int(r["prediction"][c]) for r in unique]
        scores = [float(r["scores"][c]) for r in unique]
        print(
            f"{c:<9}  {sum(predictions):>18}  {sum(v > 0 for v in scores):>13}  "
            f"{min(scores):>9.3f}  {sum(scores)/len(scores):>10.3f}  {max(scores):>9.3f}"
        )
    all_zero = all(
        all(int(r["prediction"][c]) == 0 for c in base.CRITERIA)
        for r in unique
    )
    print("All-zero collapse:", all_zero)
    if len(unique) < 8:
        print("Decision: INCOMPLETE — rerun failed or missing samples")
    elif all_zero:
        print("Decision: FAIL — do not run the formal 30-image evaluation")
    else:
        print("Decision: PROMPT ACTIVE — inspect per-criterion errors before freezing formal run")


def parse_wrapper() -> argparse.Namespace:
    if len(sys.argv) > 1 and sys.argv[1] == "make-dev-manifest":
        p = argparse.ArgumentParser(description="Create disjoint 8-combination development manifest")
        p.add_argument("command")
        p.add_argument("--source", type=Path, default=Path("manifests/test_791_seed42.csv"))
        p.add_argument("--formal-manifest", type=Path, default=Path("manifests/test_30_stratified_seed42.csv"))
        p.add_argument("--replacement-manifest", type=Path, default=Path("manifests/replacement_011_content_filter.csv"))
        p.add_argument("--output", type=Path, default=Path("manifests/dev8_nonanchored_v2_seed20260720.csv"))
        p.add_argument("--seed", type=int, default=DEV_SEED)
        ns = p.parse_args()
        ns.func = make_dev_manifest
        return ns
    if len(sys.argv) > 1 and sys.argv[1] == "development-report":
        p = argparse.ArgumentParser(description="Check whether an 8-image v2 run still collapses")
        p.add_argument("command")
        p.add_argument("--results", type=Path, required=True)
        ns = p.parse_args()
        ns.func = development_report
        return ns

    # Reuse the tested v1 CLI and replace only the run callback.
    base.DIRECT_PROMPT = DIRECT_PROMPT_V2
    base.soc_prompt = soc_prompt_v2
    base.run = run_v2
    return base.parser().parse_args()


if __name__ == "__main__":
    try:
        namespace = parse_wrapper()
        namespace.func(namespace)
    except KeyboardInterrupt:
        print("Interrupted; successful JSONL rows are preserved.", file=sys.stderr)
        raise SystemExit(130)


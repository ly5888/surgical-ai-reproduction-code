#!/usr/bin/env python3
"""Reproducible 30-image comparison for Qwen3-VL-Plus Direct vs SoC-no-FS.

Run from ~/sumofchecks_reproduction after activating .venv.
The script reads DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL and DASHSCOPE_MODEL
from .env. It never prints the API key.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CRITERIA = {
    "C1": {
        "definition": "Exactly two and only two tubular structures visibly enter the gallbladder.",
        "checks": [
            ("R1", "The gallbladder neck or infundibulum is visible.", 0.10),
            ("R2", "Exactly two tubular structures enter the gallbladder.", 0.30),
            ("R3", "Both structures have visible continuity with the gallbladder.", 0.20),
            ("R4", "No additional third tubular structure enters the gallbladder.", 0.20),
            ("R5", "The two structures are clearly separated and distinguishable.", 0.10),
            ("R6", "Relevant anatomy is not obstructed.", 0.10),
        ],
    },
    "C2": {
        "definition": "The hepatocystic triangle is cleared of fat and fibrous tissue.",
        "checks": [
            ("R1", "The hepatocystic triangle is visible.", 0.10),
            ("R2", "Fat and fibrous tissue are cleared from the triangle.", 0.30),
            ("R3", "The cystic duct and cystic artery are exposed within the triangle.", 0.20),
            ("R4", "Underlying liver parenchyma is visible through the dissected area.", 0.15),
            ("R5", "Exposure is sufficient to assess clearance.", 0.15),
            ("R6", "The triangle is not obstructed.", 0.10),
        ],
    },
    "C3": {
        "definition": "The lower third of the gallbladder is detached from the liver bed.",
        "checks": [
            ("R1", "The gallbladder and liver-bed interface are visible.", 0.10),
            ("R2", "The lower third of the gallbladder is assessable.", 0.10),
            ("R3", "The lower third is separated from the liver bed.", 0.25),
            ("R4", "The cystic plate or liver surface is exposed behind it.", 0.20),
            ("R5", "A clear dissection plane is visible.", 0.15),
            ("R6", "The view is sufficient to judge detachment.", 0.10),
            ("R7", "The interface is not obstructed.", 0.10),
        ],
    },
}

DIRECT_PROMPT = """Assess the Critical View of Safety (CVS) in this single laparoscopic cholecystectomy frame.

C1: Exactly two and only two tubular structures visibly enter the gallbladder.
C2: The hepatocystic triangle is cleared of fat and fibrous tissue.
C3: The lower third of the gallbladder is detached from the liver bed.

Assign 1 only when a criterion is clearly demonstrated in this frame. Assign 0 when it is absent, incomplete, obscured, or uncertain.
Return one valid JSON object only, without markdown or commentary, exactly in this schema:
{"C1":0,"C2":0,"C3":0}
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(row: dict[str, Any]) -> str:
    return "".join(str(int(float(row[c]))) for c in ("c1", "c2", "c3"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocate_strata(sizes: dict[str, int], n: int, minimum: int = 2) -> dict[str, int]:
    """Minimum representation, then capacity-proportional largest remainder."""
    if n > sum(sizes.values()):
        raise ValueError(f"Requested {n} rows but source contains only {sum(sizes.values())}")
    allocation = {k: 0 for k in sorted(sizes)}
    remaining = n
    # Give each non-empty label combination up to `minimum` examples, round-robin.
    for _ in range(minimum):
        for k in sorted(sizes):
            if remaining and allocation[k] < sizes[k]:
                allocation[k] += 1
                remaining -= 1
    while remaining:
        capacity = {k: sizes[k] - allocation[k] for k in sizes}
        total_capacity = sum(capacity.values())
        quotas = {k: remaining * capacity[k] / total_capacity for k in sizes}
        floors = {k: min(capacity[k], math.floor(quotas[k])) for k in sizes}
        added = sum(floors.values())
        for k in sizes:
            allocation[k] += floors[k]
        remaining -= added
        if not remaining:
            break
        order = sorted(
            sizes,
            key=lambda k: (-(quotas[k] - floors[k]), k),
        )
        progressed = False
        for k in order:
            if remaining and allocation[k] < sizes[k]:
                allocation[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Could not complete stratum allocation")
    return allocation


def make_manifest(args: argparse.Namespace) -> None:
    rows = read_csv(args.source)
    required = {"image_path", "c1", "c2", "c3"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Source CSV must contain: {sorted(required)}")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[label(row)].append(row)
    sizes = {k: len(v) for k, v in groups.items()}
    allocation = allocate_strata(sizes, args.n, args.minimum_per_stratum)
    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        population = sorted(groups[key], key=lambda r: (r.get("sample_index", ""), r["image_path"]))
        selected.extend(rng.sample(population, allocation[key]))
    selected.sort(key=lambda r: int(r.get("sample_index", 0)))
    for i, row in enumerate(selected):
        row["stratified_index"] = str(i)
        row["label_combo"] = label(row)
    # Put the two audit columns first, retaining every source column.
    ordered = []
    for row in selected:
        ordered.append({"stratified_index": row.pop("stratified_index"), "label_combo": row.pop("label_combo"), **row})
    write_csv(args.output, ordered)
    print(f"Manifest: {args.output}")
    print(f"Rows: {len(ordered)}  Seed: {args.seed}  SHA256: {sha256(args.output)}")
    print("Source distribution:", dict(sorted(sizes.items())))
    print("Selected allocation:", dict(sorted(allocation.items())))


def extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                obj, _ = decoder.raw_decode(text[match.start():])
                return obj
            except json.JSONDecodeError:
                continue
        raise


def parse_direct(text: str) -> dict[str, int]:
    obj = extract_json(text)
    result = {}
    for c in ("C1", "C2", "C3"):
        value = obj.get(c, obj.get(c.lower()))
        if isinstance(value, bool):
            value = int(value)
        value = int(value)
        if value not in (0, 1):
            raise ValueError(f"{c} must be 0 or 1, got {value!r}")
        result[c] = value
    return result


def parse_checks(text: str, expected: Iterable[str]) -> dict[str, dict[str, str]]:
    obj = extract_json(text)
    if "checks" in obj and isinstance(obj["checks"], list):
        obj = {str(x.get("id")): x for x in obj["checks"]}
    result = {}
    for rid in expected:
        item = obj.get(rid)
        if not isinstance(item, dict):
            raise ValueError(f"Missing object for {rid}")
        judgment = str(item.get("judgment", "")).strip().lower()
        if judgment not in {"yes", "no", "uncertain"}:
            raise ValueError(f"Invalid judgment for {rid}: {judgment!r}")
        result[rid] = {"judgment": judgment, "reason": str(item.get("reason", "")).strip()}
    return result


def data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def load_client(args: argparse.Namespace):
    try:
        from dotenv import dotenv_values
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("Install dependencies: pip install openai python-dotenv") from e
    cfg = dotenv_values(args.env)
    api_key = cfg.get("DASHSCOPE_API_KEY")
    base_url = cfg.get("DASHSCOPE_BASE_URL")
    model = args.model or cfg.get("DASHSCOPE_MODEL") or "qwen3-vl-plus"
    if not api_key or not base_url:
        raise SystemExit(".env must contain DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout), model


def call_model(client: Any, model: str, image: str, prompt: str, args: argparse.Namespace):
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": prompt},
                ]}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body={"enable_thinking": False},
            )
            return response
        except Exception as e:  # API and network exceptions vary by SDK version.
            last_error = e
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8))
    raise last_error  # type: ignore[misc]


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def direct_one(client: Any, model: str, image: str, args: argparse.Namespace) -> dict[str, Any]:
    response = call_model(client, model, image, DIRECT_PROMPT, args)
    raw = response.choices[0].message.content
    pred = parse_direct(raw)
    return {"prediction": pred, "scores": {c: float(pred[c]) for c in pred},
            "raw": raw, "calls": [{"criterion": "all", "usage": usage_dict(response)}]}


def soc_prompt(criterion: str) -> str:
    spec = CRITERIA[criterion]
    checks = "\n".join(f'{rid}: {text}' for rid, text, _ in spec["checks"])
    example = ",".join(f'"{rid}":{{"judgment":"no","reason":"short reason"}}'
                       for rid, _, _ in spec["checks"])
    return f"""Evaluate {criterion} in this single laparoscopic cholecystectomy frame.

{criterion} is satisfied only when: {spec['definition']}

Checks:
{checks}

For each check use yes only with clear visible evidence, no when absent, and uncertain when visibility is insufficient. Judge each check independently.
Return one valid JSON object only. Do not use markdown or trailing commas. Include every check key exactly once:
{{{example}}}
"""


def soc_one(client: Any, model: str, image: str, args: argparse.Namespace) -> dict[str, Any]:
    predictions, scores, details, calls = {}, {}, {}, []
    for c, spec in CRITERIA.items():
        response = call_model(client, model, image, soc_prompt(c), args)
        raw = response.choices[0].message.content
        expected = [rid for rid, _, _ in spec["checks"]]
        judgments = parse_checks(raw, expected)
        score = sum(weight for rid, _, weight in spec["checks"]
                    if judgments[rid]["judgment"] == "yes")
        score = round(score, 6)
        scores[c] = score
        predictions[c] = int(score > 0.5)
        details[c] = {"checks": judgments, "raw": raw}
        calls.append({"criterion": c, "usage": usage_dict(response)})
    return {"prediction": predictions, "scores": scores, "details": details, "calls": calls}


def completed_keys(path: Path) -> set[str]:
    result = set()
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if not row.get("error") and row.get("prediction"):
                    result.add(str(row["stratified_index"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return result


def run(args: argparse.Namespace) -> None:
    rows = read_csv(args.manifest)
    if not rows:
        raise ValueError("Manifest is empty")
    done = completed_keys(args.output)
    pending = [r for r in rows if str(r.get("stratified_index", r.get("sample_index"))) not in done]
    if args.max_new is not None:
        pending = pending[:args.max_new]
    if args.dry_run:
        print(f"Method={args.method}; total={len(rows)}; completed={len(done)}; pending this run={len(pending)}")
        for row in pending[:5]:
            print(row.get("stratified_index"), row["image_path"], label(row), Path(row["image_path"]).exists())
        return
    client, model = load_client(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Method={args.method}  Model={model}  completed={len(done)}  pending={len(pending)}")
    for number, row in enumerate(pending, 1):
        key = str(row.get("stratified_index", row.get("sample_index")))
        record: dict[str, Any] = {
            "stratified_index": key,
            "sample_index": row.get("sample_index"),
            "file_name": row.get("file_name") or Path(row["image_path"]).name,
            "image_path": row["image_path"],
            "truth": {c.upper(): int(float(row[c])) for c in ("c1", "c2", "c3")},
            "method": args.method,
            "model": model,
            "temperature": args.temperature,
            "threshold": 0.5 if args.method == "soc-nofs" else None,
            "error": None,
        }
        try:
            image = data_url(Path(row["image_path"]))
            result = direct_one(client, model, image, args) if args.method == "direct" else soc_one(client, model, image, args)
            record.update(result)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        pred = record.get("prediction")
        print(f"[{number}/{len(pending)}] {record['file_name']} truth={label(row)} pred={pred or 'ERROR'}")


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("error") and row.get("prediction"):
                latest[str(row["stratified_index"])] = row
    return latest


def safe_div(a: int, b: int) -> float | None:
    return a / b if b else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows), "exact": 0, "criteria": {}}
    for row in rows:
        out["exact"] += int(all(row["truth"][c] == row["prediction"][c] for c in CRITERIA))
    out["exact_accuracy"] = safe_div(out["exact"], len(rows))
    for c in CRITERIA:
        tp = sum(r["truth"][c] == 1 and r["prediction"][c] == 1 for r in rows)
        tn = sum(r["truth"][c] == 0 and r["prediction"][c] == 0 for r in rows)
        fp = sum(r["truth"][c] == 0 and r["prediction"][c] == 1 for r in rows)
        fn = sum(r["truth"][c] == 1 and r["prediction"][c] == 0 for r in rows)
        precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out["criteria"][c] = {
            "accuracy": safe_div(tp + tn, len(rows)), "precision": precision,
            "recall": recall, "specificity": safe_div(tn, tn + fp), "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
    out["total_tokens"] = sum(
        call.get("usage", {}).get("total_tokens") or 0 for row in rows for call in row.get("calls", [])
    )
    out["api_calls"] = sum(len(row.get("calls", [])) for row in rows)
    return out


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def summarize(args: argparse.Namespace) -> None:
    direct, soc = load_latest(args.direct), load_latest(args.soc)
    common = sorted(set(direct) & set(soc), key=int)
    if not common:
        raise ValueError("No completed samples common to both result files")
    dm, sm = metrics([direct[k] for k in common]), metrics([soc[k] for k in common])
    report = {
        "common_completed_samples": len(common),
        "direct": dm,
        "soc_nofs": sm,
        "notes": [
            "SoC checks and weights are reconstructed because the authors did not release exact prompts/checklists.",
            "This is a deterministic pipeline comparison, not an exact reproduction of the paper's hidden implementation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Common completed samples: {len(common)}")
    print("metric             direct    soc-no-fs")
    print(f"exact accuracy     {fmt(dm['exact_accuracy']):>7}    {fmt(sm['exact_accuracy']):>9}")
    for c in CRITERIA:
        for name in ("accuracy", "precision", "recall", "specificity", "f1"):
            print(f"{c} {name:<11} {fmt(dm['criteria'][c][name]):>7}    {fmt(sm['criteria'][c][name]):>9}")
    print(f"API calls          {dm['api_calls']:>7}    {sm['api_calls']:>9}")
    print(f"Total tokens       {dm['total_tokens']:>7}    {sm['total_tokens']:>9}")
    print(f"Report: {args.output}")


def self_test(_: argparse.Namespace) -> None:
    sizes = {"000": 20, "001": 5, "010": 3, "111": 2}
    allocation = allocate_strata(sizes, 12, 2)
    assert sum(allocation.values()) == 12
    assert all(0 <= allocation[k] <= sizes[k] for k in sizes)
    assert parse_direct('```json\n{"C1": 1, "C2": 0, "C3": 1}\n```') == {"C1": 1, "C2": 0, "C3": 1}
    checks = parse_checks('prefix {"R1":{"judgment":"yes","reason":"x"}} suffix', ["R1"])
    assert checks["R1"]["judgment"] == "yes"
    assert abs(sum(w for _, _, w in CRITERIA["C1"]["checks"]) - 1.0) < 1e-9
    assert abs(sum(w for _, _, w in CRITERIA["C2"]["checks"]) - 1.0) < 1e-9
    assert abs(sum(w for _, _, w in CRITERIA["C3"]["checks"]) - 1.0) < 1e-9
    print("Self-test passed")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make-manifest", help="Create deterministic label-combination-stratified manifest")
    m.add_argument("--source", type=Path, default=Path("manifests/test_791_seed42.csv"))
    m.add_argument("--output", type=Path, default=Path("manifests/test_30_stratified_seed42.csv"))
    m.add_argument("--n", type=int, default=30)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--minimum-per-stratum", type=int, default=2)
    m.set_defaults(func=make_manifest)
    r = sub.add_parser("run", help="Run one method with resumable JSONL output")
    r.add_argument("--method", required=True, choices=("direct", "soc-nofs"))
    r.add_argument("--manifest", type=Path, default=Path("manifests/test_30_stratified_seed42.csv"))
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--env", type=Path, default=Path(".env"))
    r.add_argument("--model", default=None)
    r.add_argument("--temperature", type=float, default=0.1)
    r.add_argument("--max-tokens", type=int, default=500)
    r.add_argument("--timeout", type=float, default=120)
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--max-new", type=int, default=None, help="Process at most N new rows, useful for smoke tests")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=run)
    s = sub.add_parser("summarize", help="Compare completed samples common to both JSONL files")
    s.add_argument("--direct", type=Path, required=True)
    s.add_argument("--soc", type=Path, required=True)
    s.add_argument("--output", type=Path, default=Path("results/stratified30_comparison.json"))
    s.set_defaults(func=summarize)
    t = sub.add_parser("self-test")
    t.set_defaults(func=self_test)
    return p


if __name__ == "__main__":
    try:
        ns = parser().parse_args()
        ns.func(ns)
    except KeyboardInterrupt:
        print("Interrupted; completed JSONL rows are preserved. Re-run the same command to resume.", file=sys.stderr)
        raise SystemExit(130)

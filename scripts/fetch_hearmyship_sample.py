#!/usr/bin/env python3
"""Fetch a stratified, group-split WAV sample from https://hearmyship.fer.hr/."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from download_hearmyship import download, make_url  # noqa: E402

ORIGIN = "https://hearmyship.fer.hr"
MEDIA_PREFIX = "data/sounds-and-images/"
DEFAULT_JSON = ROOT / "data/external/hearmyship/metadata/vessels.json"
DEFAULT_SPLIT = ROOT / "data/external/hearmyship/metadata/hearmyship_split_v1.csv"
DEFAULT_AUDIO_DIR = ROOT / "data/external/hearmyship/audio"
USER_AGENT = "Research acoustic downloader (https://hearmyship.fer.hr/); Python requests"


def _is_allowed_url(url: str) -> bool:
    return url == ORIGIN or url.startswith(f"{ORIGIN}/")


def _parse_valid_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    with path.open("rb") as fh:
        head = fh.read(12)
    return head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE"


def normalize_media_relpath(raw_sound: str) -> str:
    """Normalize HearMyShip sound field to data/sounds-and-images/... relative POSIX path."""
    rel = raw_sound.replace("\\", "/").lstrip("/")
    if not rel.startswith(MEDIA_PREFIX):
        rel = MEDIA_PREFIX + rel
    return rel


def discover_vessels_json(session: requests.Session) -> tuple[str, list[dict]]:
    """Fetch complete HearMyShip vessel metadata from https://hearmyship.fer.hr/api/."""
    dl_url = f"{ORIGIN}/api/vessels-download"
    resp = session.get(dl_url, timeout=(10, 30), allow_redirects=True)
    if _is_allowed_url(resp.url) and resp.status_code == 200:
        payload = resp.json()
        if isinstance(payload, list) and any(
            isinstance(r, dict) and isinstance(r.get("sound"), str) and r["sound"].lower().endswith(".wav")
            for r in payload
        ):
            return dl_url, payload

    # Fallback: paginate /api/vessels?page=N (which always includes sound relative paths)
    records: list[dict] = []
    first = session.get(f"{ORIGIN}/api/vessels?page=1", timeout=(10, 30), allow_redirects=True)
    first.raise_for_status()
    if not _is_allowed_url(first.url):
        raise RuntimeError(f"Unexpected redirect: {first.url}")
    first_json = first.json()
    records.extend(first_json.get("vessels", []))
    total_pages = int(first_json.get("totalPages", 1))
    for page in range(2, total_pages + 1):
        r = session.get(f"{ORIGIN}/api/vessels?page={page}", timeout=(10, 30), allow_redirects=True)
        r.raise_for_status()
        if not _is_allowed_url(r.url):
            raise RuntimeError(f"Unexpected redirect: {r.url}")
        records.extend(r.json().get("vessels", []))
        time.sleep(0.08)

    if not records:
        raise RuntimeError(f"No vessel records retrieved from {ORIGIN}/api/vessels")
    return f"{ORIGIN}/api/vessels?page=1..{total_pages}", records


def extract_vessel_group(record: dict) -> str:
    """Derive vessel/passage group key from relative path so same-passage clips stay in one split."""
    rel = normalize_media_relpath(str(record["sound"]))
    parts = Path(rel).parts
    folder = parts[-2] if len(parts) >= 2 else Path(rel).stem
    category = str(record.get("category") or "unknown").strip() or "unknown"
    return f"{category}::{folder}"


def select_stratified_sample(
    records: list[dict],
    audio_dir: Path,
    target_count: int,
    seed: int,
) -> list[dict]:
    """Select target_count unique-path WAVs stratified across all categories, keeping local valid WAVs."""
    valid_by_rel: dict[str, dict] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        sound = rec.get("sound")
        if not isinstance(sound, str) or not sound.lower().endswith(".wav"):
            continue
        rel = normalize_media_relpath(sound)
        if ".." in Path(rel).parts:
            continue
        entry = {
            "id": str(rec.get("id", rel)),
            "category": str(rec.get("category") or "unknown").strip() or "unknown",
            "relative_sound": sound,
            "rel_posix": rel,
            "vessel_group": extract_vessel_group(rec),
        }
        valid_by_rel.setdefault(rel, entry)

    valid = list(valid_by_rel.values())
    if not valid:
        raise ValueError("No valid WAV records found in vessels.json")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    existing: list[dict] = []
    for item in valid:
        dest = audio_dir / item["rel_posix"]
        item["already_exists"] = _parse_valid_wav(dest)
        if item["already_exists"]:
            existing.append(item)
        by_cat[item["category"]].append(item)

    rng = random.Random(seed)
    chosen_rels: set[str] = {x["rel_posix"] for x in existing}
    chosen: list[dict] = list(existing)

    cats = sorted(by_cat)
    pools: dict[str, list[dict]] = {}
    for cat in cats:
        rem = [x for x in by_cat[cat] if x["rel_posix"] not in chosen_rels]
        rng.shuffle(rem)
        pools[cat] = rem

    while len(chosen) < min(target_count, len(valid)):
        progressed = False
        for cat in cats:
            if len(chosen) >= min(target_count, len(valid)):
                break
            if pools[cat]:
                item = pools[cat].pop()
                chosen_rels.add(item["rel_posix"])
                chosen.append(item)
                progressed = True
        if not progressed:
            break

    return chosen


def assign_group_splits(items: list[dict], val_frac: float = 0.2, seed: int = 42) -> list[dict]:
    """Assign deterministic 80/20 group split by vessel_group within each category."""
    rng = random.Random(seed)
    by_cat_groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_cat_groups[item["category"]][item["vessel_group"]].append(item)

    for cat in sorted(by_cat_groups):
        groups = sorted(by_cat_groups[cat])
        rng.shuffle(groups)
        n_val = max(1, round(len(groups) * val_frac)) if len(groups) >= 2 else 0
        val_groups = set(groups[:n_val])
        for grp, grp_items in by_cat_groups[cat].items():
            split = "validation" if grp in val_groups else "train"
            for item in grp_items:
                item["source_split"] = split

    train_groups = {x["vessel_group"] for x in items if x["source_split"] == "train"}
    val_groups = {x["vessel_group"] for x in items if x["source_split"] == "validation"}
    overlap = train_groups & val_groups
    assert not overlap, f"Group leakage detected: {overlap}"
    return items


def run_self_test() -> None:
    """Validate path normalization, stratified sampling, and zero group-leakage splits."""
    assert normalize_media_relpath(r"\Yachts\g1\a.wav") == "data/sounds-and-images/Yachts/g1/a.wav"
    assert normalize_media_relpath(r"\data\sounds-and-images\Yachts\g1\a.wav") == "data/sounds-and-images/Yachts/g1/a.wav"
    synthetic = [
        {"id": str(i), "category": ["Yacht", "Ferry", "Tug"][i % 3], "sound": f"\\Cat\\g{i // 2}\\c{i}.wav"}
        for i in range(60)
    ]
    sample = select_stratified_sample(synthetic, Path("/nonexistent"), target_count=30, seed=42)
    assert len(sample) == 30
    counts = defaultdict(int)
    for row in sample:
        counts[row["category"]] += 1
    assert set(counts.keys()) == {"Yacht", "Ferry", "Tug"}
    assert all(v == 10 for v in counts.values())
    split = assign_group_splits(sample, val_frac=0.2, seed=42)
    tr_g = {r["vessel_group"] for r in split if r["source_split"] == "train"}
    va_g = {r["vessel_group"] for r in split if r["source_split"] == "validation"}
    assert not (tr_g & va_g)
    print("fetch_hearmyship_sample self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--split-out", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--target-count", type=int, default=120)
    parser.add_argument("--delay", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return
    if args.delay < 1.0:
        parser.error("--delay must be >= 1.0 second")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.split_out.parent.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,audio/wav,*/*;q=0.5"})
        if args.json.is_file():
            records = json.loads(args.json.read_text(encoding="utf-8"))
            json_url = "local_cache"
        else:
            json_url, records = discover_vessels_json(session)
            args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")

        selected = select_stratified_sample(records, args.audio_dir, args.target_count, args.seed)
        selected = assign_group_splits(selected, val_frac=0.2, seed=args.seed)
        print(
            json.dumps(
                {
                    "json_source": json_url,
                    "total_json_records": len(records),
                    "selected_count": len(selected),
                    "categories": sorted({x["category"] for x in selected}),
                },
                indent=2,
            ),
            flush=True,
        )

        rows_out = []
        ok_count = 0
        attempted_download = False
        for idx, item in enumerate(selected, 1):
            url = make_url(item["rel_posix"], ORIGIN)
            dest = args.audio_dir / item["rel_posix"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if _parse_valid_wav(dest):
                status = "EXISTS"
                nbytes = dest.stat().st_size
            else:
                status, nbytes = download(session, url, dest)
                if not attempted_download:
                    attempted_download = True
                    if status != "OK":
                        print(f"First remote download failed ({status}) for {url}", flush=True)
                        rows_out.append(
                            {
                                "id": item["id"],
                                "category": item["category"],
                                "vessel_group": item["vessel_group"],
                                "relative_sound": item["relative_sound"],
                                "url": url,
                                "local_path": str(dest.relative_to(ROOT)),
                                "source_split": item["source_split"],
                                "status": status,
                                "bytes": nbytes,
                            }
                        )
                        break
                if idx < len(selected):
                    time.sleep(args.delay + random.uniform(0.0, 0.25))
            if status in ("OK", "EXISTS"):
                ok_count += 1
            rows_out.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "vessel_group": item["vessel_group"],
                    "relative_sound": item["relative_sound"],
                    "url": url,
                    "local_path": str(dest.relative_to(ROOT)),
                    "source_split": item["source_split"],
                    "status": status,
                    "bytes": nbytes,
                }
            )
            print(f"[{idx}/{len(selected)}] {item['category']} id={item['id']} -> {status} ({nbytes} B)", flush=True)

    with args.split_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "category", "vessel_group", "relative_sound", "url", "local_path", "source_split", "status", "bytes"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Completed: {ok_count}/{len(rows_out)} valid WAVs. Manifest: {args.split_out}")
    if ok_count < 20:
        raise SystemExit(f"Insufficient HearMyShip WAVs downloaded ({ok_count} < 20)")


if __name__ == "__main__":
    main()

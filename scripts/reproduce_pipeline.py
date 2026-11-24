#!/usr/bin/env python3
"""Master deterministic reproducibility runner for Modeling v2, v3, v4 (with v5 extension hook) and Acoustic Standards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "reports/reproducibility_manifest.json"
SUPPORTED_VERSIONS = ("v2", "v3", "v4")

SELF_TEST_CMDS = [
    ("v2_train", ["scripts/modeling_v2_train.py", "--self-test-only"]),
    ("v3_train", ["scripts/modeling_v3_train.py", "--self-test-only"]),
    ("v4_simulator", ["scripts/modeling_v4_simulator.py", "--self-test-only"]),
    ("v4_features", ["scripts/modeling_v4_features.py", "--self-test-only"]),
    ("v4_train", ["scripts/modeling_v4_train.py", "--self-test-only"]),
    ("standards_suite", ["scripts/standardized_evaluation_suite.py", "--self-test-only"]),
    ("sa_site_activity", ["scripts/evaluate_sa_relative_vessel_activity.py", "--self-test-only"]),
]

TRACKED_ARTIFACTS = [
    "requirements.txt",
    "data/processed/vtuad_decidecade_2s_minutes.parquet",
    "data/processed/hearmyship_decidecade_2s_clips.parquet",
    "data/synthetic/modeling_v3/manifest.json",
    "data/synthetic/modeling_v4/manifest.json",
    "submissions/candidate_v2_ensemble.csv",
    "submissions/candidate_v3_ensemble.csv",
    "submissions/candidate_v4_ensemble.csv",
    "reports/standards_benchmark/standardized_metrics_full.csv",
    "reports/standards_benchmark/sa_site_relative_vessel_activity.csv",
    "reports/standards_benchmark/sa_heldout_signal_support.csv",
    "reports/standards_benchmark/summary.json",
]

DELIVERABLE_ARTIFACTS = [
    "deliverables/overleaf_manuscript/main.tex",
    "deliverables/overleaf_manuscript/references.bib",
    "deliverables/overleaf_manuscript/research_manuscript_latex.pdf",
    "deliverables/overleaf_manuscript/overleaf_research_manuscript.zip",
    "deliverables/overleaf_manuscript/research_and_modeling_code_final.zip",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _det_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "42",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/mpl",
    })
    return env


def _run_py(cmd_args: list[str], label: str) -> str:
    t0 = time.perf_counter()
    res = subprocess.run(
        [sys.executable, *cmd_args],
        cwd=ROOT,
        env=_det_env(),
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(
            f"Step '{label}' failed (rc={res.returncode}):\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}"
        )
    print(f"[PASS] {label} ({dt:.2f}s)")
    return res.stdout.strip()


def _check_requirements_compliance() -> dict:
    req_path = ROOT / "requirements.txt"
    lines = [ln.strip() for ln in req_path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    banned_automl = ("autogluon", "flaml", "h2o", "tpot", "pycaret", "auto-sklearn", "autosklearn", "evalml", "mljar")
    hits = [ln for ln in lines if any(b in ln.lower() for b in banned_automl)]
    assert not hits, f"Prohibited AutoML dependency in requirements.txt: {hits}"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pinned_packages": len(lines),
        "requirements_sha256": _sha256(req_path),
        "automl_check": "ZERO_AUTOML_PASS",
    }


def _check_submissions() -> list[dict]:
    sample = pd.read_csv(ROOT / "data/zindi/raw/sample_submission.csv")
    rows = []
    for ver, rel in [
        ("v2", "submissions/candidate_v2_ensemble.csv"),
        ("v3", "submissions/candidate_v3_ensemble.csv"),
        ("v4", "submissions/candidate_v4_ensemble.csv"),
    ]:
        p = ROOT / rel
        assert p.is_file(), f"Missing submission candidate: {p}"
        df = pd.read_csv(p)
        assert list(df.columns) == list(sample.columns), f"Schema mismatch in {rel}"
        assert len(df) == len(sample) == 1500, f"Row count mismatch in {rel}"
        assert (df["ID"] == sample["ID"]).all(), f"ID mismatch in {rel}"
        assert np.isfinite(df["share_added_vessel"]).all() and df["share_added_vessel"].between(0.0, 1.0).all()
        assert np.isfinite(df["score_added_vessel"]).all() and df["score_added_vessel"].between(0.0, 1.0).all()
        rows.append({
            "version": ver,
            "path": rel,
            "sha256": _sha256(p),
            "share_mean": float(df["share_added_vessel"].mean()),
            "score_mean": float(df["score_added_vessel"].mean()),
        })
    print(f"[PASS] All {len(rows)} submission CSVs match sample_submission.csv schema & bounds")
    return rows


def run_verify_only(write_manifest: bool = False) -> dict:
    env_info = _check_requirements_compliance()
    self_test_outputs = {}
    for name, cmd in SELF_TEST_CMDS:
        out = _run_py(cmd, f"self-test:{name}")
        assert "PASS" in out, f"Self-test '{name}' did not print PASS: {out}"
        self_test_outputs[name] = out.splitlines()[-1] if out else "PASS"

    sub_checks = _check_submissions()

    checksums = {}
    for rel in TRACKED_ARTIFACTS + DELIVERABLE_ARTIFACTS:
        p = ROOT / rel
        if p.is_file():
            checksums[rel] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}
        elif rel in TRACKED_ARTIFACTS:
            raise FileNotFoundError(f"Missing required artifact: {rel}")

    payload = {
        "schema_version": 1,
        "supported_versions": list(SUPPORTED_VERSIONS),
        "environment": env_info,
        "deterministic_flags": {
            "PYTHONHASHSEED": "42",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "self_tests": self_test_outputs,
        "submissions": sub_checks,
        "checksums": checksums,
    }

    if write_manifest:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[PASS] Wrote {MANIFEST_PATH.relative_to(ROOT)} ({len(checksums)} checksums)")
    elif MANIFEST_PATH.is_file():
        saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for rel, meta in saved.get("checksums", {}).items():
            if rel in checksums:
                assert checksums[rel]["sha256"] == meta["sha256"], (
                    f"Checksum drift detected for {rel}: current={checksums[rel]['sha256']} vs manifest={meta['sha256']}"
                )
        print(f"[PASS] Verified against {MANIFEST_PATH.relative_to(ROOT)}")

    return payload


def run_export_only(version: str) -> None:
    targets = SUPPORTED_VERSIONS if version == "all" else (version,)
    if "v4" in targets:
        pre_sha = _sha256(ROOT / "submissions/candidate_v4_ensemble.csv")
        _run_py(["scripts/modeling_v4_train.py", "--export-only"], "export-only:v4")
        post_sha = _sha256(ROOT / "submissions/candidate_v4_ensemble.csv")
        assert pre_sha == post_sha, f"v4 export-only altered CSV hash: {pre_sha} vs {post_sha}"
    _run_py(["scripts/standardized_evaluation_suite.py"], "export-only:standards_suite")
    _check_submissions()


def run_full(version: str) -> None:
    targets = SUPPORTED_VERSIONS if version == "all" else (version,)
    if "v2" in targets:
        _run_py(["scripts/modeling_v2_simulator.py"], "full:v2_simulator")
        _run_py(["scripts/modeling_v2_train.py"], "full:v2_train")
    if "v3" in targets:
        _run_py(["scripts/modeling_v3_simulator.py"], "full:v3_simulator")
        _run_py(["scripts/modeling_v3_train.py"], "full:v3_train")
    if "v4" in targets:
        _run_py(["scripts/modeling_v4_simulator.py"], "full:v4_simulator")
        _run_py(["scripts/modeling_v4_train.py"], "full:v4_train")
    if "v4" in targets or "v3" in targets or "v2" in targets:
        _run_py(["scripts/standardized_evaluation_suite.py"], "full:standards_suite")
        _run_py(["scripts/build_modeling_notebook_03.py"], "full:build_nb03")
        _run_py(["scripts/build_overleaf_manuscript.py"], "full:overleaf_assets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["verify-only", "export-only", "full"], default="verify-only")
    parser.add_argument("--version", choices=["v2", "v3", "v4", "all"], default="all")
    parser.add_argument("--write-manifest", action="store_true", help="Write/update reports/reproducibility_manifest.json.")
    args = parser.parse_args()

    if args.mode == "verify-only":
        run_verify_only(write_manifest=args.write_manifest)
    elif args.mode == "export-only":
        run_export_only(args.version)
    else:
        run_full(args.version)
        run_verify_only(write_manifest=args.write_manifest)


if __name__ == "__main__":
    main()

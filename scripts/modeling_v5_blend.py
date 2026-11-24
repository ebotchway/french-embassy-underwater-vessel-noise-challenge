#!/usr/bin/env python3
"""Zero-tuning 50/50 ensemble combining Modeling v4 (magnitude) and v5-simple (h(f)-invariant presence)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

W_V4_DEFAULT = 0.50
SUBMISSION_DEFAULT = ROOT / "submissions/candidate_v5_blend.csv"
REPORTS_DIR_DEFAULT = ROOT / "reports/modeling_v5_blend"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blend_predictions_v5(
    share_v4: np.ndarray,
    score_v4: np.ndarray,
    share_v5: np.ndarray,
    score_v5: np.ndarray,
    w_v4: float = W_V4_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-independent fixed convex blend of v4 (broadband magnitude) and v5_simple (h(f)-invariant detection)."""
    if not (0.0 <= float(w_v4) <= 1.0):
        raise ValueError(f"w_v4 must lie in [0, 1], got {w_v4}")
    sh4 = np.asarray(share_v4, dtype=np.float64)
    sc4 = np.asarray(score_v4, dtype=np.float64)
    sh5 = np.asarray(share_v5, dtype=np.float64)
    sc5 = np.asarray(score_v5, dtype=np.float64)
    if not (sh4.shape == sc4.shape == sh5.shape == sc5.shape):
        raise ValueError(f"Mismatched prediction shapes: {sh4.shape}, {sc4.shape}, {sh5.shape}, {sc5.shape}")

    w4 = float(w_v4)
    w5 = 1.0 - w4
    share_blend = np.clip(w4 * sh4 + w5 * sh5, 0.0, 1.0)
    score_blend = np.clip(w4 * sc4 + w5 * sc5, 0.0, 1.0)
    return share_blend, score_blend


def run_self_test() -> None:
    """Verify exact 50/50 blending, [0, 1] clipping, row-independence, and algebraic MSE diversity identity."""
    rng = np.random.default_rng(42)
    y = rng.uniform(0.0, 1.0, size=64)
    sh4 = rng.uniform(0.0, 1.0, size=64)
    sh5 = rng.uniform(0.0, 1.0, size=64)
    sc4 = rng.uniform(0.0, 1.0, size=64)
    sc5 = rng.uniform(0.0, 1.0, size=64)

    sh_b, sc_b = blend_predictions_v5(sh4, sc4, sh5, sc5, w_v4=0.50)
    assert np.allclose(sh_b, 0.5 * (sh4 + sh5), atol=1e-15, rtol=0.0)
    assert np.allclose(sc_b, 0.5 * (sc4 + sc5), atol=1e-15, rtol=0.0)

    # Algebraic MSE diversity identity: MSE(0.5*(a+b)) = 0.5*MSE(a) + 0.5*MSE(b) - 0.25*mean((a-b)^2)
    mse4 = float(np.mean((y - sh4) ** 2))
    mse5 = float(np.mean((y - sh5) ** 2))
    mse_b = float(np.mean((y - sh_b) ** 2))
    expected_mse_b = 0.5 * mse4 + 0.5 * mse5 - 0.25 * float(np.mean((sh4 - sh5) ** 2))
    mse_err = abs(mse_b - expected_mse_b)
    assert mse_err <= 1e-14, f"MSE diversity identity mismatch: {mse_err}"

    # Permutation and slice row-independence + out-of-range clipping
    perm = rng.permutation(64)
    sh_p, sc_p = blend_predictions_v5(sh4[perm], sc4[perm], sh5[perm], sc5[perm], w_v4=0.50)
    assert np.array_equal(sh_b[perm], sh_p) and np.array_equal(sc_b[perm], sc_p)
    sh_sub, sc_sub = blend_predictions_v5(sh4[:10], sc4[:10], sh5[:10], sc5[:10], w_v4=0.50)
    assert np.array_equal(sh_b[:10], sh_sub) and np.array_equal(sc_b[:10], sc_sub)

    sh_clip, sc_clip = blend_predictions_v5(
        np.array([-0.2, 1.3]), np.array([-0.1, 1.2]),
        np.array([-0.2, 1.1]), np.array([0.0, 1.1]),
        w_v4=0.50,
    )
    assert np.array_equal(sh_clip, np.array([0.0, 1.0]))
    assert np.array_equal(sc_clip, np.array([0.0, 1.0]))
    print(f"PASS modeling_v5_blend self-test (50/50 blend, row-independent, mse_id_err={mse_err:.2e})")


def _eval_split(y: np.ndarray, sh: np.ndarray, sc: np.ndarray, split: str, model: str) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        mean_absolute_error,
        roc_auc_score,
        root_mean_squared_error,
    )

    yb = (y > 0.0).astype(int)
    z0 = y == 0.0
    rmse = float(root_mean_squared_error(y, sh))
    return {
        "split": split,
        "model": model,
        "n": int(len(y)),
        "zindi_1_minus_rmse": float(1.0 - rmse),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, sh)),
        "zero_rmse": float(root_mean_squared_error(y[z0], sh[z0])) if z0.any() else np.nan,
        "roc_auc": float(roc_auc_score(yb, sc)) if len(np.unique(yb)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(yb, sc)) if yb.any() else np.nan,
        "logloss": float(log_loss(yb, np.clip(sc, 1e-6, 1.0 - 1e-6))) if len(np.unique(yb)) == 2 else np.nan,
        "brier": float(brier_score_loss(yb, sc)) if len(np.unique(yb)) == 2 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--w-v4", type=float, default=W_V4_DEFAULT)
    parser.add_argument("--submission-out", type=Path, default=SUBMISSION_DEFAULT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR_DEFAULT)
    args = parser.parse_args()

    run_self_test()
    if args.self_test_only:
        return

    t0 = time.perf_counter()
    # Import torch before GBDT / sklearn bundles on macOS
    import torch  # noqa: F401
    import joblib
    from modeling_v4_simulator import LOG10_F_NORM
    from modeling_v4_train import predict_from_bundle_v4
    from modeling_v5_simple import TEST_COLS, predict_from_bundle_v5_simple

    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading saved bundle_v4 and bundle_v5_simple...", flush=True)
    bundle_v4 = joblib.load(ROOT / "models/v4/bundle_v4.joblib")
    bundle_v5 = joblib.load(ROOT / "models/v5_simple/bundle_v5_simple.joblib")

    print("[2/3] Evaluating v4, v5_simple, and 50/50 v5_blend across validation, holdout, +8 dB/dec stress, and real SA strata...", flush=True)
    rows = []
    cached = {}
    tilt8 = (8.0 * LOG10_F_NORM)[None, None, :]

    z_val_cal = np.load(ROOT / "data/synthetic/modeling_v4/val_cal_hard.npz")
    z_val_std = np.load(ROOT / "data/synthetic/modeling_v4/val_standard.npz")
    z_hold = np.load(ROOT / "data/synthetic/modeling_v4/holdout.npz")

    split_specs = [
        ("val_cal_hard", z_val_cal["X"].astype(np.float64), z_val_cal["y"].astype(np.float64)),
        ("val_standard", z_val_std["X"].astype(np.float64), z_val_std["y"].astype(np.float64)),
        ("holdout", z_hold["X"].astype(np.float64), z_hold["y"].astype(np.float64)),
        ("holdout_tilt_+8dB_dec", z_hold["X"].astype(np.float64) + tilt8, z_hold["y"].astype(np.float64)),
    ]

    for sp_name, X_arr, y_arr in split_specs:
        sh4, sc4 = predict_from_bundle_v4(bundle_v4, X_arr)
        sh5, sc5 = predict_from_bundle_v5_simple(bundle_v5, X_arr)
        sh_b, sc_b = blend_predictions_v5(sh4, sc4, sh5, sc5, w_v4=args.w_v4)
        cached[sp_name] = {
            "y": y_arr,
            "sh4": sh4,
            "sc4": sc4,
            "sh5": sh5,
            "sc5": sc5,
            "sh_b": sh_b,
            "sc_b": sc_b,
        }
        rows.append(_eval_split(y_arr, sh4, sc4, sp_name, "Ensemble_v4"))
        rows.append(_eval_split(y_arr, sh5, sc5, sp_name, "Ensemble_v5_Simple"))
        rows.append(_eval_split(y_arr, sh_b, sc_b, sp_name, "Ensemble_v5_Blend"))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.reports_dir / "comparison_metrics.csv", index=False)

    # Real unmixed SA (y == 0) response across cal-hard (all 5 strata incl. strict_hf_ship) and unseen holdout
    def _stratum_summary(sp_key: str, prov_rel: str) -> dict:
        prov = pd.read_csv(ROOT / prov_rel)
        d = cached[sp_key]
        m0 = d["y"] == 0.0
        df0 = prov.loc[m0, ["station", "bg_stratum"]].copy()
        df0["v4_share"], df0["v4_score"] = d["sh4"][m0], d["sc4"][m0]
        df0["v5_share"], df0["v5_score"] = d["sh5"][m0], d["sc5"][m0]
        df0["blend_share"], df0["blend_score"] = d["sh_b"][m0], d["sc_b"][m0]
        cols = ["v4_share", "v4_score", "v5_share", "v5_score", "blend_share", "blend_score"]
        return df0.groupby("bg_stratum")[cols].mean().round(6).to_dict("index")

    real_sa_cal_hard_y0 = _stratum_summary("val_cal_hard", "data/synthetic/modeling_v4/val_cal_hard_provenance.csv")
    real_sa_holdout_y0 = _stratum_summary("holdout", "data/synthetic/modeling_v4/holdout_provenance.csv")

    print("[3/3] Generating strictly inference-only Test.csv blend and cross-checking saved submissions...", flush=True)
    test_df = pd.read_csv(ROOT / "data/zindi/raw/Test.csv")
    sample_sub = pd.read_csv(ROOT / "data/zindi/raw/sample_submission.csv")
    assert (test_df["ID"] == sample_sub["ID"]).all()
    X_test = test_df[TEST_COLS].to_numpy(dtype=np.float64).reshape(-1, 15, 25)

    sh4_t, sc4_t = predict_from_bundle_v4(bundle_v4, X_test)
    sh5_t, sc5_t = predict_from_bundle_v5_simple(bundle_v5, X_test)

    # Verify bundle predictions match checked-in submissions exactly
    sub_v4_csv = pd.read_csv(ROOT / "submissions/candidate_v4_ensemble.csv")
    sub_v5_csv = pd.read_csv(ROOT / "submissions/candidate_v5_simple.csv")
    max_diff_v4_sh = float(np.max(np.abs(sh4_t - sub_v4_csv["share_added_vessel"].to_numpy())))
    max_diff_v4_sc = float(np.max(np.abs(sc4_t - sub_v4_csv["score_added_vessel"].to_numpy())))
    max_diff_v5_sh = float(np.max(np.abs(sh5_t - sub_v5_csv["share_added_vessel"].to_numpy())))
    max_diff_v5_sc = float(np.max(np.abs(sc5_t - sub_v5_csv["score_added_vessel"].to_numpy())))

    sh_blend_t, sc_blend_t = blend_predictions_v5(sh4_t, sc4_t, sh5_t, sc5_t, w_v4=args.w_v4)
    sh_blend_csv, sc_blend_csv = blend_predictions_v5(
        sub_v4_csv["share_added_vessel"].to_numpy(),
        sub_v4_csv["score_added_vessel"].to_numpy(),
        sub_v5_csv["share_added_vessel"].to_numpy(),
        sub_v5_csv["score_added_vessel"].to_numpy(),
        w_v4=args.w_v4,
    )
    max_diff_csv_sh = float(np.max(np.abs(sh_blend_t - sh_blend_csv)))
    max_diff_csv_sc = float(np.max(np.abs(sc_blend_t - sc_blend_csv)))
    # CNN2D_v4 uses float32 forward passes (~1e-8 precision across runs/batch sizes)
    assert max_diff_csv_sh <= 1e-6 and max_diff_csv_sc <= 1e-6, (
        f"Bundle-vs-CSV blend mismatch: share={max_diff_csv_sh}, score={max_diff_csv_sc}"
    )

    sub_out = pd.DataFrame({
        "ID": sample_sub["ID"],
        "share_added_vessel": sh_blend_t,
        "score_added_vessel": sc_blend_t,
    })
    assert list(sub_out.columns) == list(sample_sub.columns) and len(sub_out) == 1500
    assert np.isfinite(sh_blend_t).all() and np.isfinite(sc_blend_t).all()
    assert sub_out["share_added_vessel"].between(0.0, 1.0).all()
    assert sub_out["score_added_vessel"].between(0.0, 1.0).all()
    sub_out.to_csv(args.submission_out, index=False)

    summary = {
        "elapsed_s": float(time.perf_counter() - t0),
        "weights": {"v4": float(args.w_v4), "v5_simple": float(1.0 - args.w_v4)},
        "submission_path": str(args.submission_out.relative_to(ROOT)),
        "submission_sha256": _sha256(args.submission_out),
        "cross_check_max_diff": {
            "v4_share": max_diff_v4_sh,
            "v4_score": max_diff_v4_sc,
            "v5_share": max_diff_v5_sh,
            "v5_score": max_diff_v5_sc,
            "blend_csv_vs_bundle_share": max_diff_csv_sh,
            "blend_csv_vs_bundle_score": max_diff_csv_sc,
        },
        "test_stats": {
            "share_mean": float(sh_blend_t.mean()),
            "share_median": float(np.median(sh_blend_t)),
            "share_std": float(sh_blend_t.std()),
            "share_q10_q50_q90": [float(np.quantile(sh_blend_t, q)) for q in (0.1, 0.5, 0.9)],
            "score_mean": float(sc_blend_t.mean()),
            "score_median": float(np.median(sc_blend_t)),
            "score_std": float(sc_blend_t.std()),
            "score_gt_090_frac": float((sc_blend_t > 0.90).mean()),
            "score_gt_095_frac": float((sc_blend_t > 0.95).mean()),
            "v4_vs_v5_share_corr": float(np.corrcoef(sh4_t, sh5_t)[0, 1]),
            "v4_vs_v5_score_corr": float(np.corrcoef(sc4_t, sc5_t)[0, 1]),
            "v4_vs_v5_share_rms_diff": float(np.sqrt(np.mean((sh4_t - sh5_t) ** 2))),
        },
        "real_sa_val_cal_hard_y0_strata": real_sa_cal_hard_y0,
        "real_sa_holdout_y0_strata": real_sa_holdout_y0,
    }
    (args.reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== v4 vs v5_simple vs v5_blend (50/50) Evaluation ===")
    print(metrics_df.round(4).to_string(index=False))
    print("\n=== Test.csv Blend Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

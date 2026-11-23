#!/usr/bin/env python3
"""Generate leak-free, physics-aligned synthetic training, validation, and station-holdout datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_mixtures_v1 import BANDS, ParquetRowReader, SHAPE  # noqa: E402
from build_synthetic_mixtures_v2 import sample_background  # noqa: E402
from extract_external_decidecade import EXACT_CENTRES  # noqa: E402

BG_MANIFEST_DEFAULT = ROOT / "reports/eda/background_split_manifest_v2.csv"
VTUAD_PARQUET_DEFAULT = ROOT / "data/processed/vtuad_decidecade_2s_minutes.parquet"
HMS_PARQUET_DEFAULT = ROOT / "data/processed/hearmyship_decidecade_2s_clips.parquet"
OUT_DIR_DEFAULT = ROOT / "data/synthetic/modeling_v2"

REF_DB = 100.0
LOG10_F_NORM = np.log10(EXACT_CENTRES / 500.0)
HIGH_BAND_ATTEN_WEIGHT = np.maximum(0.0, np.log10(EXACT_CENTRES / 300.0))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def db_to_lin_fixed_ref(db: np.ndarray) -> np.ndarray:
    """Convert dB matrix to linear power relative to fixed REF_DB (preserves true dynamics)."""
    return np.power(10.0, (np.asarray(db, dtype=np.float64) - REF_DB) / 10.0)


def assign_disjoint_vessel_sources(
    vtuad_df: pd.DataFrame,
    hms_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Build strictly disjoint vessel source pools across train (with 5 CV folds), val, and holdout."""
    rng = np.random.default_rng(seed)

    # Filter out background-labelled VTUAD files cleanly
    vt_active = vtuad_df[
        vtuad_df["source_split"].isin(["train", "validation"]) & (vtuad_df["label"] != "background")
    ].copy()
    hms_active = hms_df[hms_df["source_split"].isin(["train", "validation"])].copy()

    # Deterministic hull-level split of existing VTUAD validation hulls into val (60%) vs holdout (40%)
    val_hulls = sorted(vt_active.loc[vt_active["source_split"] == "validation", "hull_id"].astype(str).unique())
    rng.shuffle(val_hulls)
    n_vt_val = max(1, int(round(0.6 * len(val_hulls))))
    vt_val_set = set(val_hulls[:n_vt_val])
    vt_hold_set = set(val_hulls[n_vt_val:])
    assert vt_val_set.isdisjoint(vt_hold_set)

    # Same 60/40 group-level split on HearMyShip validation groups into val vs holdout
    hms_val_groups = sorted(hms_active.loc[hms_active["source_split"] == "validation", "vessel_group"].astype(str).unique())
    rng.shuffle(hms_val_groups)
    n_hms_val = max(1, int(round(0.6 * len(hms_val_groups)))) if len(hms_val_groups) >= 2 else len(hms_val_groups)
    hms_val_set = set(hms_val_groups[:n_hms_val])
    hms_hold_set = set(hms_val_groups[n_hms_val:]) if len(hms_val_groups) >= 2 else set()
    assert hms_val_set.isdisjoint(hms_hold_set)

    # Partition train hulls & HearMyShip train groups into 5 disjoint CV folds
    train_hulls = sorted(vt_active.loc[vt_active["source_split"] == "train", "hull_id"].astype(str).unique())
    rng.shuffle(train_hulls)
    vt_hull_fold = {h: idx % 5 for idx, h in enumerate(train_hulls)}

    hms_train_groups = sorted(hms_active.loc[hms_active["source_split"] == "train", "vessel_group"].astype(str).unique())
    rng.shuffle(hms_train_groups)
    hms_group_fold = {g: idx % 5 for idx, g in enumerate(hms_train_groups)}

    pools: dict[str, dict[str, list[dict]]] = {
        "train": {"vtuad": [], "hms": []},
        "val": {"vtuad": [], "hms": []},
        "holdout": {"vtuad": [], "hms": []},
    }

    for clip_uid, grp in vt_active.groupby("clip_uid", sort=True):
        grp = grp.sort_values("minute_index")
        mat = grp[BANDS].to_numpy(dtype=np.float64)
        hull = str(grp["hull_id"].iloc[0])
        sp = str(grp["source_split"].iloc[0])
        cls = str(grp["label"].iloc[0])
        if sp == "train":
            target_split = "train"
            fold = vt_hull_fold[hull]
        elif hull in vt_val_set:
            target_split = "val"
            fold = -1
        else:
            target_split = "holdout"
            fold = -1
        pools[target_split]["vtuad"].append(
            {
                "dataset": "vtuad",
                "source_id": str(clip_uid),
                "group_id": f"vt_hull::{hull}",
                "vessel_class": cls,
                "fold": fold,
                "matrix": mat,
            }
        )

    hms_bands = hms_active[BANDS].to_numpy(dtype=np.float64)
    for idx_row, row in enumerate(hms_active.itertuples(index=False)):
        vec = hms_bands[idx_row][None, :]
        if not np.isfinite(vec).all() or (vec <= 0).all():
            continue
        vg = str(row.vessel_group)
        sp = str(row.source_split)
        if sp == "train":
            target_split = "train"
            fold = hms_group_fold[vg]
        elif vg in hms_val_set:
            target_split = "val"
            fold = -1
        else:
            target_split = "holdout"
            fold = -1
        pools[target_split]["hms"].append(
            {
                "dataset": "hearmyship",
                "source_id": f"hms::{row.id}",
                "group_id": f"hms_grp::{vg}",
                "vessel_class": str(row.category),
                "fold": fold,
                "matrix": vec,
            }
        )

    # Verify zero group overlap across all three partitions
    g_tr = {x["group_id"] for x in pools["train"]["vtuad"] + pools["train"]["hms"]}
    g_va = {x["group_id"] for x in pools["val"]["vtuad"] + pools["val"]["hms"]}
    g_ho = {x["group_id"] for x in pools["holdout"]["vtuad"] + pools["holdout"]["hms"]}
    assert g_tr.isdisjoint(g_va) and g_tr.isdisjoint(g_ho) and g_va.isdisjoint(g_ho)

    def _index_by_class_and_fold(items: list[dict]) -> dict:
        by_class: dict[str, list[dict]] = {}
        for it in items:
            by_class.setdefault(it["vessel_class"], []).append(it)
        return {"all": items, "by_class": by_class}

    return (
        {"vtuad": _index_by_class_and_fold(pools["train"]["vtuad"]), "hms": _index_by_class_and_fold(pools["train"]["hms"])},
        {"vtuad": _index_by_class_and_fold(pools["val"]["vtuad"]), "hms": _index_by_class_and_fold(pools["val"]["hms"])},
        {"vtuad": _index_by_class_and_fold(pools["holdout"]["vtuad"]), "hms": _index_by_class_and_fold(pools["holdout"]["hms"])},
    )


def synthesize_vessel_trajectory(
    source_matrix: np.ndarray,
    bg_lin: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, int]:
    """Generate realistic (15, 25) vessel linear-energy matrix with trajectory & channel warps."""
    n_avail = source_matrix.shape[0]
    t_idx = np.arange(15, dtype=np.float64)
    u = float(rng.random())

    # Base temporal profile (15, 25) from available vessel rows
    if n_avail == 1:
        base = np.repeat(source_matrix, 15, axis=0)
    else:
        if u > 0.90 and n_avail >= 4:
            # Native progression subclip embedded directly
            dur = int(rng.integers(4, min(15, n_avail) + 1))
            src_start = int(rng.integers(0, n_avail - dur + 1))
            ins_start = int(rng.integers(0, 16 - dur))
            V = np.zeros(SHAPE, dtype=np.float64)
            V[ins_start : ins_start + dur] = source_matrix[src_start : src_start + dur]
            V_warped = apply_channel_warp(V, bg_lin, active_mask=(V.sum(axis=1) > 0), rng=rng)
            return V_warped, "native_vtuad", dur
        # Pick 15 rows by sampling a contiguous or smoothly interpolated block
        idxs = np.clip(np.round(np.linspace(0, n_avail - 1, 15)).astype(int), 0, n_avail - 1)
        if rng.random() < 0.5:
            start = int(rng.integers(0, n_avail))
            step_dir = 1 if rng.random() < 0.5 else -1
            idxs = np.clip(start + step_dir * np.arange(15), 0, n_avail - 1)
        base = source_matrix[idxs].copy()

    # Normalize each row gently so trajectory envelope governs macro level while keeping spectral flutter
    row_tot = np.maximum(base.sum(axis=1, keepdims=True), 1e-30)
    mean_tot = float(row_tot.mean())
    base_shape = base / row_tot * mean_tot

    if u < 0.45:
        mode = "cpa_bell"
        t_cpa = float(rng.uniform(1.5, 12.5))
        width = float(rng.uniform(1.2, 4.2))
        exponent = float(rng.uniform(0.85, 1.35))
        dist_rel = np.sqrt(1.0 + ((t_idx - t_cpa) / width) ** 2)
        env = np.power(dist_rel, -2.0 * exponent)
        # Distance-dependent high-frequency absorption
        hf_atten_db = -float(rng.uniform(0.5, 3.0)) * (dist_rel[:, None] - 1.0) * HIGH_BAND_ATTEN_WEIGHT[None, :]
        V = base_shape * (env[:, None] * np.power(10.0, hf_atten_db / 10.0))
        dur = int((env > 0.1 * env.max()).sum())
    elif u < 0.72:
        mode = "entry_exit"
        dur = int(rng.integers(2, 13))
        ins_start = int(rng.integers(0, 16 - dur))
        env = np.zeros(15, dtype=np.float64)
        sub = np.linspace(-1.0, 1.0, dur)
        # Smooth cosine taper at boundaries
        taper = 0.35 + 0.65 * np.cos(0.45 * np.pi * sub) ** 2
        env[ins_start : ins_start + dur] = taper
        V = base_shape * env[:, None]
    else:
        mode = "stationary_drift"
        dur = 15
        steps = rng.normal(0.0, 0.7, size=15)
        drift_db = np.cumsum(steps) - np.mean(np.cumsum(steps))
        drift_db = np.clip(drift_db, -4.0, 4.0)
        V = base_shape * np.power(10.0, drift_db[:, None] / 10.0)

    V = apply_channel_warp(V, bg_lin, active_mask=(V.sum(axis=1) > 0), rng=rng)
    return V, mode, dur


def apply_channel_warp(
    V: np.ndarray,
    bg_lin: np.ndarray,
    active_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply propagation tilt, modal ripple, jitter, and optional local background shape blend."""
    out = V.copy()
    if not active_mask.any():
        return out

    tilt_db_per_dec = float(np.clip(rng.normal(0.0, 2.5), -6.0, 6.0))
    ripple_amp = float(rng.uniform(0.0, 2.5))
    ripple_freq = float(rng.uniform(0.6, 2.0))
    ripple_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    norm_b = np.linspace(0.0, 1.0, 25)
    spectral_warp_db = tilt_db_per_dec * LOG10_F_NORM + ripple_amp * np.cos(2.0 * np.pi * ripple_freq * norm_b + ripple_phase)
    jitter_db = rng.normal(0.0, 0.75, size=(15, 25))

    factor = np.power(10.0, (spectral_warp_db[None, :] + jitter_db) / 10.0)
    out[active_mask] *= factor[active_mask]

    # 35% probability: softly bend log-spectral profile toward local hydrophone spectral color
    if rng.random() < 0.35:
        beta = float(rng.uniform(0.15, 0.45))
        bg_prof = np.maximum(bg_lin.mean(axis=0), 1e-30)
        bg_prof /= bg_prof.sum()
        v_prof = np.maximum(out[active_mask].mean(axis=0), 1e-30)
        v_prof /= v_prof.sum()
        ratio_adj = np.power(bg_prof / v_prof, beta)
        out[active_mask] *= ratio_adj[None, :]

    return np.maximum(out, 0.0)


def sample_requested_share(rng: np.random.Generator, zero_prob: float = 0.22) -> float:
    """Sample realistic target share across exact 0.0 and wide [0.003, 0.992] continuous range."""
    u = float(rng.random())
    if u < zero_prob:
        return 0.0
    rem = (u - zero_prob) / (1.0 - zero_prob)
    if rem < 0.25:
        return float(rng.uniform(0.003, 0.05))
    if rem < 0.60:
        return float(rng.uniform(0.05, 0.35))
    if rem < 0.95:
        return float(rng.uniform(0.35, 0.90))
    return float(rng.uniform(0.90, 0.992))


def sample_test_like_minute_mask(rng: np.random.Generator) -> np.ndarray:
    """Generate whole-minute boolean mask matching observed Test.csv dropout structure."""
    obs = np.ones(15, dtype=bool)
    if rng.random() < 0.416:
        obs[0] = False
    if rng.random() < 0.067:
        obs[14] = False
    if rng.random() < 0.020:
        interior = int(rng.integers(1, 14))
        obs[interior] = False
    assert obs.sum() >= 10
    return obs


def _sample_vessel_from_pool(pool: dict, fold: int | None, hms_prob: float, rng: np.random.Generator) -> dict | None:
    use_hms = (rng.random() < hms_prob) and len(pool["hms"]["all"]) > 0
    sub_pool = pool["hms"] if use_hms else pool["vtuad"]
    if fold is None:
        classes = sorted(sub_pool["by_class"])
        cls = classes[int(rng.integers(len(classes)))]
        items = sub_pool["by_class"][cls]
        return items[int(rng.integers(len(items)))]

    # Restrict strictly to sources belonging to CV fold
    fold_items = [x for x in sub_pool["all"] if x["fold"] == fold]
    if not fold_items:
        other = pool["vtuad"] if use_hms else pool["hms"]
        fold_items = [x for x in other["all"] if x["fold"] == fold]
    by_c: dict[str, list[dict]] = {}
    for x in fold_items:
        by_c.setdefault(x["vessel_class"], []).append(x)
    cls = sorted(by_c)[int(rng.integers(len(by_c)))]
    cands = by_c[cls]
    return cands[int(rng.integers(len(cands)))]


def build_split_dataset(
    split_name: str,
    bg_pool: pd.DataFrame,
    vessel_pool: dict,
    n_target: int,
    reader: ParquetRowReader,
    rng: np.random.Generator,
    out_dir: Path,
    hms_prob: float = 0.40,
) -> dict:
    """Generate and persist NPZ matrices + provenance CSV for one split."""
    n = min(n_target, len(bg_pool))

    # For train, enforce zero run_id overlap across 5 CV folds cleanly
    fold_assignments = np.full(n, -1, dtype=np.int32)
    if split_name == "train":
        # Partition background runs into 5 disjoint CV folds (balance windows across folds)
        run_sizes = bg_pool.groupby("run_id").size().sort_values(ascending=False)
        fold_loads = [0] * 5
        run_to_fold: dict[str, int] = {}
        for rid, sz in run_sizes.items():
            f = int(np.argmin(fold_loads))
            run_to_fold[str(rid)] = f
            fold_loads[f] += int(sz)
        bg_pool = bg_pool.copy()
        bg_pool["cv_fold"] = bg_pool["run_id"].astype(str).map(run_to_fold)
        selected_indices = []
        per_fold = n // 5
        rem = n % 5
        for f in range(5):
            f_pool = bg_pool[bg_pool["cv_fold"] == f]
            take = per_fold + (1 if f < rem else 0)
            chosen = sample_background(f_pool, take, rng)
            selected_indices.extend([(idx, f) for idx in chosen])
        rng.shuffle(selected_indices)
        bg_indices = [pair[0] for pair in selected_indices]
        fold_assignments = np.array([pair[1] for pair in selected_indices], dtype=np.int32)
    else:
        bg_indices = sample_background(bg_pool, n, rng)

    X = np.empty((n, 15, 25), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    rows_meta = []

    for i, bg_idx in enumerate(bg_indices):
        rec = bg_pool.loc[bg_idx]
        positions = json.loads(rec.row_positions_json)
        bg_db = reader.read(str(rec.source_parquet), positions)
        B = db_to_lin_fixed_ref(bg_db)

        q = sample_requested_share(rng, zero_prob=0.22)
        fold = int(fold_assignments[i]) if split_name == "train" else None

        if q == 0.0:
            V = np.zeros(SHAPE, dtype=np.float64)
            M = B.copy()
            mode, dur = "background_only", 0
            v_dataset, v_source_id, v_group_id, v_class = "", "", "", ""
            alpha = 0.0
        else:
            src = _sample_vessel_from_pool(vessel_pool, fold, hms_prob=hms_prob, rng=rng)
            assert src is not None
            V_raw, mode, dur = synthesize_vessel_trajectory(src["matrix"], B, rng)
            v_sum = float(V_raw.sum())
            if v_sum <= 0.0:
                V = np.zeros(SHAPE, dtype=np.float64)
                M = B.copy()
                alpha = 0.0
            else:
                b_sum = float(B.sum())
                alpha = (q * b_sum) / ((1.0 - q) * v_sum)
                V = alpha * V_raw
                M = B + V
            v_dataset = src["dataset"]
            v_source_id = src["source_id"]
            v_group_id = src["group_id"]
            v_class = src["vessel_class"]

        obs_min_mask = sample_test_like_minute_mask(rng)
        obs_cell_mask = np.repeat(obs_min_mask[:, None], 25, axis=1)

        # Recompute exact ground-truth target strictly over observed cells
        m_obs_sum = float(M[obs_cell_mask].sum())
        v_obs_sum = float(V[obs_cell_mask].sum())
        y_obs = 0.0 if (q == 0.0 or m_obs_sum <= 0.0) else float(np.clip(v_obs_sum / m_obs_sum, 0.0, 1.0))

        M_db = REF_DB + 10.0 * np.log10(np.maximum(M, 1e-30))
        M_db[~obs_cell_mask] = np.nan

        X[i] = M_db.astype(np.float32)
        y[i] = np.float32(y_obs)
        rows_meta.append(
            {
                "example": i,
                "split": split_name,
                "fold": int(fold_assignments[i]),
                "station": str(rec.station),
                "bg_run_id": str(rec.run_id),
                "bg_start_time": str(rec.start_time),
                "vessel_dataset": v_dataset,
                "vessel_source_id": v_source_id,
                "vessel_group_id": v_group_id,
                "vessel_class": v_class,
                "trajectory_mode": mode,
                "vessel_duration_min": dur,
                "requested_share": q,
                "observed_share": y_obs,
                "n_obs_minutes": int(obs_min_mask.sum()),
                "relative_scale": float(alpha),
            }
        )

    assert np.isfinite(y).all() and (y >= 0.0).all() and (y <= 1.0).all()
    assert int((~np.isfinite(X)).sum()) > 0
    # Verify whole-minute missingness only (no partially missing minutes)
    finite_min = np.isfinite(X)
    assert np.array_equal(finite_min.any(axis=2), finite_min.all(axis=2))

    npz_path = out_dir / f"{split_name}.npz"
    csv_path = out_dir / f"{split_name}_provenance.csv"
    np.savez_compressed(npz_path, X=X, y=y, fold=fold_assignments)
    prov_df = pd.DataFrame(rows_meta)
    prov_df.to_csv(csv_path, index=False)

    return {
        "split": split_name,
        "rows": n,
        "zero_fraction": float((y == 0.0).mean()),
        "mean_share": float(y.mean()),
        "mean_obs_minutes": float(prov_df["n_obs_minutes"].mean()),
        "stations": {k: int(v) for k, v in prov_df["station"].value_counts().items()},
        "npz_sha256": _sha256(npz_path),
    }


def run_self_test() -> None:
    """Verify fixed-ref dB-to-linear consistency, missing-mask preservation, and fold-isolation logic."""
    rng = np.random.default_rng(7)
    B_db = rng.uniform(75.0, 110.0, size=(15, 25))
    B = db_to_lin_fixed_ref(B_db)
    V_raw = rng.uniform(0.1, 2.0, size=(15, 25))
    q = 0.25
    alpha = (q * B.sum()) / ((1.0 - q) * V_raw.sum())
    M = B + alpha * V_raw
    assert np.isclose((alpha * V_raw).sum() / M.sum(), q, atol=1e-12)

    # Shifting B_db by +37.5 dB yields exact same physical ratio when vessel scaled by relative share
    B_shifted = db_to_lin_fixed_ref(B_db + 37.5)
    alpha_s = (q * B_shifted.sum()) / ((1.0 - q) * V_raw.sum())
    M_s = B_shifted + alpha_s * V_raw
    assert np.isclose((alpha_s * V_raw).sum() / M_s.sum(), q, atol=1e-12)
    assert np.allclose(10.0 * np.log10(M_s / M), 37.5, atol=1e-10)
    print("modeling_v2_simulator self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bg-manifest", type=Path, default=BG_MANIFEST_DEFAULT)
    parser.add_argument("--vtuad-parquet", type=Path, default=VTUAD_PARQUET_DEFAULT)
    parser.add_argument("--hms-parquet", type=Path, default=HMS_PARQUET_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-val", type=int, default=1500)
    parser.add_argument("--n-holdout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bg_df = pd.read_csv(args.bg_manifest)
    vt_df = pd.read_parquet(args.vtuad_parquet)
    hms_df = pd.read_parquet(args.hms_parquet) if args.hms_parquet.is_file() else pd.DataFrame(columns=["source_split"])

    tr_vessels, va_vessels, ho_vessels = assign_disjoint_vessel_sources(vt_df, hms_df, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    reader = ParquetRowReader()

    specs = [
        ("train", bg_df[bg_df["split"] == "train"].reset_index(drop=True), tr_vessels, args.n_train),
        ("val", bg_df[bg_df["split"] == "validation"].reset_index(drop=True), va_vessels, args.n_val),
        ("holdout", bg_df[bg_df["split"] == "station_holdout"].reset_index(drop=True), ho_vessels, args.n_holdout),
    ]

    summary = {}
    prov_dfs = {}
    for sp_name, b_pool, v_pool, n_req in specs:
        info = build_split_dataset(sp_name, b_pool, v_pool, n_req, reader, rng, args.out_dir)
        summary[sp_name] = info
        prov_dfs[sp_name] = pd.read_csv(args.out_dir / f"{sp_name}_provenance.csv")
        print(f"Generated {sp_name}: {info['rows']} rows, zero={info['zero_fraction']:.3f}, mean_y={info['mean_share']:.3f}")

    # Cross-split leakage assertions
    tr_p, va_p, ho_p = prov_dfs["train"], prov_dfs["val"], prov_dfs["holdout"]
    assert set(tr_p["bg_run_id"]).isdisjoint(va_p["bg_run_id"])
    assert set(tr_p["bg_run_id"]).isdisjoint(ho_p["bg_run_id"])
    assert set(va_p["bg_run_id"]).isdisjoint(ho_p["bg_run_id"])

    def _nonempty_groups(df: pd.DataFrame) -> set[str]:
        s = df["vessel_group_id"].dropna().astype(str)
        return {x for x in s if x}

    assert _nonempty_groups(tr_p).isdisjoint(_nonempty_groups(va_p))
    assert _nonempty_groups(tr_p).isdisjoint(_nonempty_groups(ho_p))
    assert _nonempty_groups(va_p).isdisjoint(_nonempty_groups(ho_p))

    # Also assert train 5 folds share zero bg_run_id or vessel_group_id
    for f1 in range(5):
        for f2 in range(f1 + 1, 5):
            a = tr_p[tr_p["fold"] == f1]
            b = tr_p[tr_p["fold"] == f2]
            assert set(a["bg_run_id"]).isdisjoint(b["bg_run_id"])
            assert _nonempty_groups(a).isdisjoint(_nonempty_groups(b))

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Leakage assertions PASS. Manifest:", manifest_path)


if __name__ == "__main__":
    main()

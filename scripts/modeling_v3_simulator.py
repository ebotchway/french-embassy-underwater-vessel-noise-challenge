#!/usr/bin/env python3
"""Generate leak-free, hard-negative-balanced, reef-aware synthetic datasets for Modeling v3."""
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

from build_synthetic_mixtures_v1 import BANDS, SHAPE  # noqa: E402
from extract_external_decidecade import EXACT_CENTRES  # noqa: E402

BG_MANIFEST_DEFAULT = ROOT / "reports/eda/background_split_manifest_v2.csv"
VTUAD_PARQUET_DEFAULT = ROOT / "data/processed/vtuad_decidecade_2s_minutes.parquet"
HMS_PARQUET_DEFAULT = ROOT / "data/processed/hearmyship_decidecade_2s_clips.parquet"
OUT_DIR_DEFAULT = ROOT / "data/synthetic/modeling_v3"

REF_DB = 100.0
LOG10_F_NORM = np.log10(EXACT_CENTRES / 500.0)
HIGH_BAND_ATTEN_WEIGHT = np.maximum(0.0, np.log10(EXACT_CENTRES / 300.0))

STRATA_ORDER = ("hard_event", "reef_active", "glitch_drop", "ambient_normal")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def db_to_lin_fixed_ref(db: np.ndarray) -> np.ndarray:
    """Convert dB matrix to linear power relative to fixed REF_DB (preserves true dynamics)."""
    return np.power(10.0, (np.asarray(db, dtype=np.float64) - REF_DB) / 10.0)


def profile_backgrounds_by_split(bg_df: pd.DataFrame) -> pd.DataFrame:
    """Profile every indexed 15x25 background window strictly from its own acoustic values."""
    work = bg_df.copy()
    work["bg_bb_rng_db"] = 0.0
    work["bg_hf_rng_db"] = 0.0
    work["bg_mf_rng_db"] = 0.0
    work["bg_argmax_t"] = 0
    work["bg_q_d63"] = 0.0
    work["bg_drop_db"] = 0.0
    work["stratum"] = "ambient_normal"

    # Parse all JSON positions once and read each parquet strictly once sequentially
    for pq_path, grp in work.groupby("source_parquet", sort=True):
        arr = pd.read_parquet(str(pq_path), columns=BANDS).to_numpy(dtype=np.float64)
        idx_arr = grp.index.to_numpy()
        pos_mat = np.asarray([json.loads(s) for s in grp["row_positions_json"]], dtype=np.int64)
        W = arr[pos_mat]  # (M, 15, 25)
        W_med = np.median(W, axis=(1, 2), keepdims=True)
        E = np.power(10.0, np.clip(W - W_med, -60.0, 60.0) / 10.0)

        bb_db = 10.0 * np.log10(np.maximum(E.sum(axis=2), 1e-30))
        hf_db = 10.0 * np.log10(np.maximum(E[:, :, 17:25].sum(axis=2), 1e-30))
        mf_db = 10.0 * np.log10(np.maximum(E[:, :, 11:17].sum(axis=2), 1e-30))

        bb_rng = bb_db.max(axis=1) - bb_db.min(axis=1)
        hf_rng = hf_db.max(axis=1) - hf_db.min(axis=1)
        mf_rng = mf_db.max(axis=1) - mf_db.min(axis=1)
        argmax_t = np.argmax(bb_db, axis=1)
        argmin_t = np.argmin(bb_db, axis=1)
        q_rows = W[np.arange(len(W)), argmin_t, :]
        q_d63 = q_rows[:, 23] - q_rows[:, 21]
        drop_db = np.median(bb_db, axis=1) - bb_db.min(axis=1)

        is_glitch = drop_db > 14.0
        is_reef = (~is_glitch) & (q_d63 > 1.5) & (bb_rng >= 8.0)
        is_hard = (
            (~is_glitch)
            & (~is_reef)
            & (argmax_t >= 1)
            & (argmax_t <= 13)
            & ((bb_rng >= 10.0) | ((hf_rng >= 12.0) & (mf_rng >= 10.0)))
        )
        stratum = np.full(len(W), "ambient_normal", dtype=object)
        stratum[is_hard] = "hard_event"
        stratum[is_reef] = "reef_active"
        stratum[is_glitch] = "glitch_drop"

        work.loc[idx_arr, "bg_bb_rng_db"] = bb_rng
        work.loc[idx_arr, "bg_hf_rng_db"] = hf_rng
        work.loc[idx_arr, "bg_mf_rng_db"] = mf_rng
        work.loc[idx_arr, "bg_argmax_t"] = argmax_t
        work.loc[idx_arr, "bg_q_d63"] = q_d63
        work.loc[idx_arr, "bg_drop_db"] = drop_db
        work.loc[idx_arr, "stratum"] = stratum

    return work


def read_windows_fast(bg_sub: pd.DataFrame, indices: list[int]) -> np.ndarray:
    """Batch-read (N, 15, 25) background dB matrices indexed by bg_sub row index."""
    sub = bg_sub.loc[indices]
    out = np.empty((len(indices), 15, 25), dtype=np.float64)
    pos_map = {idx: out_i for out_i, idx in enumerate(indices)}
    for pq_path, grp in sub.groupby("source_parquet", sort=True):
        arr = pd.read_parquet(str(pq_path), columns=BANDS).to_numpy(dtype=np.float64)
        pos_mat = np.asarray([json.loads(s) for s in grp["row_positions_json"]], dtype=np.int64)
        W = arr[pos_mat]
        out_idx = [pos_map[idx] for idx in grp.index]
        out[out_idx] = W
    return out


def assign_disjoint_vessel_sources_v3(
    vtuad_df: pd.DataFrame,
    hms_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Build strictly disjoint vessel source pools across train (5 CV folds), val, and holdout."""
    rng = np.random.default_rng(seed)

    vt_active = vtuad_df[
        vtuad_df["source_split"].isin(["train", "validation"]) & (vtuad_df["label"] != "background")
    ].copy()
    hms_active = hms_df[hms_df["source_split"].isin(["train", "validation"])].copy()

    val_hulls = sorted(vt_active.loc[vt_active["source_split"] == "validation", "hull_id"].astype(str).unique())
    rng.shuffle(val_hulls)
    n_vt_val = max(1, int(round(0.6 * len(val_hulls))))
    vt_val_set = set(val_hulls[:n_vt_val])
    vt_hold_set = set(val_hulls[n_vt_val:])
    assert vt_val_set.isdisjoint(vt_hold_set)

    hms_val_groups = sorted(hms_active.loc[hms_active["source_split"] == "validation", "vessel_group"].astype(str).unique())
    rng.shuffle(hms_val_groups)
    n_hms_val = max(1, int(round(0.6 * len(hms_val_groups)))) if len(hms_val_groups) >= 2 else len(hms_val_groups)
    hms_val_set = set(hms_val_groups[:n_hms_val])
    hms_hold_set = set(hms_val_groups[n_hms_val:]) if len(hms_val_groups) >= 2 else set()
    assert hms_val_set.isdisjoint(hms_hold_set)

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

    has_rel_power = "relative_total_power" in vt_active.columns
    for clip_uid, grp in vt_active.groupby("clip_uid", sort=True):
        grp = grp.sort_values("minute_index")
        mat = grp[BANDS].to_numpy(dtype=np.float64)
        rel_p = (
            np.maximum(grp["relative_total_power"].to_numpy(dtype=np.float64), 1e-12)
            if has_rel_power
            else np.ones(len(grp), dtype=np.float64)
        )
        rel_p = rel_p / max(float(rel_p.mean()), 1e-12)
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
                "rel_power": rel_p,
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
                "rel_power": np.ones(1, dtype=np.float64),
            }
        )

    g_tr = {x["group_id"] for x in pools["train"]["vtuad"] + pools["train"]["hms"]}
    g_va = {x["group_id"] for x in pools["val"]["vtuad"] + pools["val"]["hms"]}
    g_ho = {x["group_id"] for x in pools["holdout"]["vtuad"] + pools["holdout"]["hms"]}
    assert g_tr.isdisjoint(g_va) and g_tr.isdisjoint(g_ho) and g_va.isdisjoint(g_ho)

    def _index_by_class(items: list[dict]) -> dict:
        by_class: dict[str, list[dict]] = {}
        for it in items:
            by_class.setdefault(it["vessel_class"], []).append(it)
        return {"all": items, "by_class": by_class}

    return (
        {"vtuad": _index_by_class(pools["train"]["vtuad"]), "hms": _index_by_class(pools["train"]["hms"])},
        {"vtuad": _index_by_class(pools["val"]["vtuad"]), "hms": _index_by_class(pools["val"]["hms"])},
        {"vtuad": _index_by_class(pools["holdout"]["vtuad"]), "hms": _index_by_class(pools["holdout"]["hms"])},
    )


def apply_channel_warp_v3(
    V: np.ndarray,
    active_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply physical propagation tilt, mild modal ripple, and cell jitter (NO background shape leak)."""
    out = V.copy()
    if not active_mask.any():
        return out

    tilt_db_per_dec = float(np.clip(rng.normal(0.0, 2.0), -5.0, 5.0))
    ripple_amp = float(rng.uniform(0.0, 1.8))
    ripple_freq = float(rng.uniform(0.6, 2.0))
    ripple_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    norm_b = np.linspace(0.0, 1.0, 25)
    spectral_warp_db = (
        tilt_db_per_dec * LOG10_F_NORM
        + ripple_amp * np.cos(2.0 * np.pi * ripple_freq * norm_b + ripple_phase)
    )
    jitter_db = rng.normal(0.0, 0.55, size=(15, 25))
    factor = np.power(10.0, (spectral_warp_db[None, :] + jitter_db) / 10.0)
    out[active_mask] *= factor[active_mask]
    return np.maximum(out, 0.0)


def synthesize_vessel_trajectory_v3(
    source_matrix: np.ndarray,
    rel_power: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, int]:
    """Generate realistic (15, 25) vessel linear-energy matrix favoring interior CPA bell passages."""
    n_avail = source_matrix.shape[0]
    t_idx = np.arange(15, dtype=np.float64)
    u = float(rng.random())

    # Native multi-minute mode preserving VTUAD minute-to-minute spectral & power trajectory
    if u >= 0.80 and u < 0.92 and n_avail >= 3:
        dur = int(rng.integers(3, min(14, n_avail) + 1))
        src_start = int(rng.integers(0, n_avail - dur + 1))
        lo = 1 if (15 - dur) >= 2 else 0
        hi = max(lo + 1, 15 - dur)
        ins_start = int(rng.integers(lo, hi))
        V = np.zeros(SHAPE, dtype=np.float64)
        V[ins_start : ins_start + dur] = (
            source_matrix[src_start : src_start + dur]
            * rel_power[src_start : src_start + dur, None]
        )
        V = apply_channel_warp_v3(V, active_mask=(V.sum(axis=1) > 0), rng=rng)
        return V, "native_multi_minute", dur

    if n_avail == 1:
        base = np.repeat(source_matrix, 15, axis=0)
    else:
        idxs = np.clip(np.round(np.linspace(0, n_avail - 1, 15)).astype(int), 0, n_avail - 1)
        if rng.random() < 0.5:
            start = int(rng.integers(0, n_avail))
            step_dir = 1 if rng.random() < 0.5 else -1
            idxs = np.clip(start + step_dir * np.arange(15), 0, n_avail - 1)
        base = source_matrix[idxs].copy()

    row_tot = np.maximum(base.sum(axis=1, keepdims=True), 1e-30)
    mean_tot = float(row_tot.mean())
    base_shape = base / row_tot * mean_tot

    if u < 0.60:
        mode = "cpa_sharp_bell"
        t_cpa = float(rng.uniform(1.5, 12.5))
        width = float(rng.uniform(0.65, 2.8))
        exponent = float(rng.uniform(0.90, 1.50))
        dist_rel = np.sqrt(1.0 + ((t_idx - t_cpa) / width) ** 2)
        env = np.power(dist_rel, -2.0 * exponent)
        hf_atten_db = -float(rng.uniform(0.8, 3.5)) * (dist_rel[:, None] - 1.0) * HIGH_BAND_ATTEN_WEIGHT[None, :]
        V = base_shape * (env[:, None] * np.power(10.0, hf_atten_db / 10.0))
        dur = int((env > 0.1 * env.max()).sum())
    elif u < 0.80:
        mode = "cpa_asym_bell"
        t_cpa = float(rng.uniform(2.0, 12.0))
        w_left = float(rng.uniform(0.7, 3.6))
        w_right = float(rng.uniform(0.7, 3.6))
        exponent = float(rng.uniform(0.90, 1.45))
        widths = np.where(t_idx <= t_cpa, w_left, w_right)
        dist_rel = np.sqrt(1.0 + ((t_idx - t_cpa) / widths) ** 2)
        env = np.power(dist_rel, -2.0 * exponent)
        hf_atten_db = -float(rng.uniform(0.8, 3.5)) * (dist_rel[:, None] - 1.0) * HIGH_BAND_ATTEN_WEIGHT[None, :]
        V = base_shape * (env[:, None] * np.power(10.0, hf_atten_db / 10.0))
        dur = int((env > 0.1 * env.max()).sum())
    else:
        mode = "entry_exit_or_slow"
        if rng.random() < 0.70:
            dur = int(rng.integers(3, 13))
            ins_start = int(rng.integers(0, 16 - dur))
            env = np.zeros(15, dtype=np.float64)
            sub = np.linspace(-1.0, 1.0, dur)
            env[ins_start : ins_start + dur] = 0.30 + 0.70 * np.cos(0.45 * np.pi * sub) ** 2
            V = base_shape * env[:, None]
        else:
            dur = 15
            steps = rng.normal(0.0, 0.55, size=15)
            drift_db = np.clip(np.cumsum(steps) - np.mean(np.cumsum(steps)), -3.5, 3.5)
            V = base_shape * np.power(10.0, drift_db[:, None] / 10.0)

    V = apply_channel_warp_v3(V, active_mask=(V.sum(axis=1) > 0), rng=rng)
    return V, mode, dur


def sample_requested_share_v3(rng: np.random.Generator, zero_prob: float = 0.42) -> float:
    """Sample target share across exact 0.0 and calibrated continuous vessel share regimes."""
    u = float(rng.random())
    if u < zero_prob:
        return 0.0
    rem = (u - zero_prob) / max(1.0 - zero_prob, 1e-9)
    if rem < 0.25:
        return float(rng.uniform(0.005, 0.08))
    if rem < 0.65:
        return float(rng.uniform(0.08, 0.35))
    if rem < 0.93:
        return float(rng.uniform(0.35, 0.75))
    return float(rng.uniform(0.75, 0.96))


def sample_minute_mask_v3(rng: np.random.Generator, stratum: str) -> np.ndarray:
    """Generate whole-minute mask reflecting reef duty-cycle startup dropouts."""
    obs = np.ones(15, dtype=bool)
    p_m01 = 0.60 if stratum == "reef_active" else 0.28
    if rng.random() < p_m01:
        obs[0] = False
    if rng.random() < 0.067:
        obs[14] = False
    if rng.random() < 0.015:
        obs[int(rng.integers(1, 14))] = False
    assert obs.sum() >= 10
    return obs


def _sample_vessel_from_pool(pool: dict, fold: int | None, hms_prob: float, rng: np.random.Generator) -> dict:
    use_hms = (rng.random() < hms_prob) and len(pool["hms"]["all"]) > 0
    sub_pool = pool["hms"] if use_hms else pool["vtuad"]
    if fold is None:
        classes = sorted(sub_pool["by_class"])
        cls = classes[int(rng.integers(len(classes)))]
        items = sub_pool["by_class"][cls]
        return items[int(rng.integers(len(items)))]

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


def _pop_from_stratum(
    stratum_queues: dict[str, list[int]],
    desired_stratum: str,
    ambient_station_queues: dict[str, list[int]],
    used_set: set[int],
) -> int:
    """Draw one unused index from desired stratum, falling back cleanly across available strata."""
    def _pop_amb() -> int | None:
        while True:
            non_empty = [s for s, sq in ambient_station_queues.items() if sq]
            if not non_empty:
                return None
            s_pick = max(non_empty, key=lambda s: len(ambient_station_queues[s]))
            idx = ambient_station_queues[s_pick].pop()
            if idx not in used_set:
                used_set.add(idx)
                return idx

    if desired_stratum == "ambient_normal":
        idx = _pop_amb()
        if idx is not None:
            return idx
    else:
        q = stratum_queues.get(desired_stratum, [])
        while q:
            idx = q.pop()
            if idx not in used_set:
                used_set.add(idx)
                return idx

    for fb in ("hard_event", "reef_active", "glitch_drop", "ambient_normal"):
        if fb == "ambient_normal":
            idx = _pop_amb()
            if idx is not None:
                return idx
        else:
            q = stratum_queues.get(fb, [])
            while q:
                idx = q.pop()
                if idx not in used_set:
                    used_set.add(idx)
                    return idx
    raise RuntimeError("Exhausted all available background windows in pool")


def _build_stratum_queues(pool_df: pd.DataFrame, rng: np.random.Generator) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    stratum_queues: dict[str, list[int]] = {}
    for st in ("hard_event", "reef_active", "glitch_drop"):
        idxs = pool_df.index[pool_df["stratum"] == st].to_numpy(dtype=np.int64, copy=True)
        rng.shuffle(idxs)
        stratum_queues[st] = idxs.tolist()
    amb_df = pool_df[pool_df["stratum"] == "ambient_normal"]
    amb_st_queues: dict[str, list[int]] = {}
    for station, grp in amb_df.groupby("station", sort=True):
        idxs = grp.index.to_numpy(dtype=np.int64, copy=True)
        rng.shuffle(idxs)
        amb_st_queues[str(station)] = idxs.tolist()
    return stratum_queues, amb_st_queues


def assign_balanced_run_folds(bg_sub: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Partition background run_ids into n_folds disjoint folds stratified by hard strata & size."""
    out = bg_sub.copy()
    run_info = []
    for rid, grp in out.groupby("run_id", sort=True):
        vc = grp["stratum"].value_counts()
        run_info.append(
            {
                "run_id": str(rid),
                "n": len(grp),
                "hard": int(vc.get("hard_event", 0)),
                "reef": int(vc.get("reef_active", 0)),
                "glitch": int(vc.get("glitch_drop", 0)),
            }
        )
    rdf = pd.DataFrame(run_info).sort_values(["glitch", "reef", "hard", "n"], ascending=False)
    fold_stats = [{"n": 0, "hard": 0, "reef": 0, "glitch": 0} for _ in range(n_folds)]
    run_to_fold: dict[str, int] = {}
    for row in rdf.itertuples(index=False):
        if row.glitch > 0:
            f = min(range(n_folds), key=lambda k: (fold_stats[k]["glitch"], fold_stats[k]["n"]))
        elif row.reef > 0:
            f = min(range(n_folds), key=lambda k: (fold_stats[k]["reef"], fold_stats[k]["n"]))
        elif row.hard > 0:
            f = min(range(n_folds), key=lambda k: (fold_stats[k]["hard"], fold_stats[k]["n"]))
        else:
            f = min(range(n_folds), key=lambda k: fold_stats[k]["n"])
        run_to_fold[row.run_id] = f
        fold_stats[f]["n"] += int(row.n)
        fold_stats[f]["hard"] += int(row.hard)
        fold_stats[f]["reef"] += int(row.reef)
        fold_stats[f]["glitch"] += int(row.glitch)
    out["cv_fold"] = out["run_id"].astype(str).map(run_to_fold)
    return out


def _pick_stratum_from_probs(probs: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=np.float64)
    p /= p.sum()
    return str(keys[int(rng.choice(len(keys), p=p))])


def build_split_v3(
    split_name: str,
    bg_pool: pd.DataFrame,
    vessel_pool: dict,
    n_target: int,
    zero_prob: float,
    neg_strata_probs: dict[str, float],
    pos_strata_probs: dict[str, float],
    rng: np.random.Generator,
    out_dir: Path,
    hms_prob: float = 0.15,
) -> tuple[dict, list[int]]:
    """Generate and persist NPZ matrices + provenance CSV for one split; returns summary and used bg indices."""
    n = min(n_target, len(bg_pool))

    chosen_records: list[tuple[int, int, float]] = []  # (bg_idx, fold, requested_share)
    used_set: set[int] = set()

    if split_name == "train":
        bg_folds = assign_balanced_run_folds(bg_pool, n_folds=5)
        per_fold = n // 5
        rem = n % 5
        for f in range(5):
            f_pool = bg_folds[bg_folds["cv_fold"] == f]
            sq, aq = _build_stratum_queues(f_pool, rng)
            take = per_fold + (1 if f < rem else 0)
            for _ in range(take):
                q = sample_requested_share_v3(rng, zero_prob=zero_prob)
                target_st = _pick_stratum_from_probs(neg_strata_probs if q == 0.0 else pos_strata_probs, rng)
                bg_idx = _pop_from_stratum(sq, target_st, aq, used_set)
                chosen_records.append((bg_idx, f, q))
        rng.shuffle(chosen_records)
    else:
        sq, aq = _build_stratum_queues(bg_pool, rng)
        for _ in range(n):
            q = sample_requested_share_v3(rng, zero_prob=zero_prob)
            target_st = _pick_stratum_from_probs(neg_strata_probs if q == 0.0 else pos_strata_probs, rng)
            bg_idx = _pop_from_stratum(sq, target_st, aq, used_set)
            chosen_records.append((bg_idx, -1, q))
        rng.shuffle(chosen_records)

    bg_indices = [r[0] for r in chosen_records]
    assert len(set(bg_indices)) == len(bg_indices), f"Duplicate background index in {split_name}"
    bg_mats = read_windows_fast(bg_pool, bg_indices)

    X = np.empty((n, 15, 25), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    fold_assignments = np.array([r[1] for r in chosen_records], dtype=np.int32)
    rows_meta = []

    for i, (bg_idx, fold_i, q) in enumerate(chosen_records):
        rec = bg_pool.loc[bg_idx]
        B = db_to_lin_fixed_ref(bg_mats[i])
        fold_arg = fold_i if split_name == "train" else None

        if q == 0.0:
            V = np.zeros(SHAPE, dtype=np.float64)
            M = B.copy()
            mode, dur = "background_only", 0
            v_dataset, v_source_id, v_group_id, v_class = "", "", "", ""
            alpha = 0.0
        else:
            src = _sample_vessel_from_pool(vessel_pool, fold_arg, hms_prob=hms_prob, rng=rng)
            V_raw, mode, dur = synthesize_vessel_trajectory_v3(src["matrix"], src["rel_power"], rng)
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

        obs_min_mask = sample_minute_mask_v3(rng, str(rec.stratum))
        obs_cell_mask = np.repeat(obs_min_mask[:, None], 25, axis=1)

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
                "fold": int(fold_i),
                "station": str(rec.station),
                "bg_stratum": str(rec.stratum),
                "bg_bb_rng_db": float(rec.bg_bb_rng_db),
                "bg_hf_rng_db": float(rec.bg_hf_rng_db),
                "bg_q_d63": float(rec.bg_q_d63),
                "bg_drop_db": float(rec.bg_drop_db),
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
    finite_min = np.isfinite(X)
    assert np.array_equal(finite_min.any(axis=2), finite_min.all(axis=2))

    npz_path = out_dir / f"{split_name}.npz"
    csv_path = out_dir / f"{split_name}_provenance.csv"
    np.savez_compressed(npz_path, X=X, y=y, fold=fold_assignments)
    prov_df = pd.DataFrame(rows_meta)
    prov_df.to_csv(csv_path, index=False)

    summary = {
        "split": split_name,
        "rows": n,
        "zero_fraction": float((y == 0.0).mean()),
        "mean_share": float(y.mean()),
        "mean_obs_minutes": float(prov_df["n_obs_minutes"].mean()),
        "strata_counts": {k: int(v) for k, v in prov_df["bg_stratum"].value_counts().items()},
        "neg_strata_counts": {k: int(v) for k, v in prov_df.loc[prov_df["observed_share"] == 0, "bg_stratum"].value_counts().items()},
        "stations": {k: int(v) for k, v in prov_df["station"].value_counts().items()},
        "npz_sha256": _sha256(npz_path),
    }
    return summary, bg_indices


def run_self_test() -> None:
    """Verify fixed-ref dB consistency, +37.5 dB shift linearity, and trajectory/mask invariants."""
    rng = np.random.default_rng(7)
    B_db = rng.uniform(75.0, 110.0, size=(15, 25))
    B = db_to_lin_fixed_ref(B_db)
    V_raw = rng.uniform(0.1, 2.0, size=(15, 25))
    q = 0.25
    alpha = (q * B.sum()) / ((1.0 - q) * V_raw.sum())
    M = B + alpha * V_raw
    assert np.isclose((alpha * V_raw).sum() / M.sum(), q, atol=1e-12)

    B_shifted = db_to_lin_fixed_ref(B_db + 37.5)
    alpha_s = (q * B_shifted.sum()) / ((1.0 - q) * V_raw.sum())
    M_s = B_shifted + alpha_s * V_raw
    assert np.isclose((alpha_s * V_raw).sum() / M_s.sum(), q, atol=1e-12)
    assert np.allclose(10.0 * np.log10(M_s / M), 37.5, atol=1e-10)

    src = rng.uniform(0.01, 1.0, size=(5, 25))
    src /= src.sum(axis=1, keepdims=True)
    rp = rng.uniform(0.5, 2.0, size=5)
    for _ in range(30):
        V, _, _ = synthesize_vessel_trajectory_v3(src, rp, rng)
        assert V.shape == SHAPE and np.isfinite(V).all() and (V >= 0.0).all()
        m_reef = sample_minute_mask_v3(rng, "reef_active")
        assert m_reef.sum() >= 10
    print("modeling_v3_simulator self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bg-manifest", type=Path, default=BG_MANIFEST_DEFAULT)
    parser.add_argument("--vtuad-parquet", type=Path, default=VTUAD_PARQUET_DEFAULT)
    parser.add_argument("--hms-parquet", type=Path, default=HMS_PARQUET_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val-std", type=int, default=2000)
    parser.add_argument("--n-val-hard", type=int, default=1500)
    parser.add_argument("--n-holdout", type=int, default=541)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Profiling South African background windows by split...", flush=True)
    bg_raw = pd.read_csv(args.bg_manifest)
    bg_all = profile_backgrounds_by_split(bg_raw)

    print("Background stratum distribution by manifest split:")
    print(pd.crosstab(bg_all["split"], bg_all["stratum"], margins=True).to_string())

    vt_df = pd.read_parquet(args.vtuad_parquet)
    hms_df = pd.read_parquet(args.hms_parquet) if args.hms_parquet.is_file() else pd.DataFrame(columns=["source_split"])
    tr_vessels, va_vessels, ho_vessels = assign_disjoint_vessel_sources_v3(vt_df, hms_df, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    neg_std = {"hard_event": 0.45, "reef_active": 0.25, "glitch_drop": 0.05, "ambient_normal": 0.25}
    pos_std = {"hard_event": 0.25, "reef_active": 0.25, "glitch_drop": 0.04, "ambient_normal": 0.46}
    neg_hard = {"hard_event": 0.55, "reef_active": 0.38, "glitch_drop": 0.07, "ambient_normal": 0.00}
    pos_hard = {"hard_event": 0.35, "reef_active": 0.35, "glitch_drop": 0.05, "ambient_normal": 0.25}

    # Build val_hard_event first from validation pool, then val_standard from disjoint validation run_ids if possible
    bg_train = bg_all[bg_all["split"] == "train"].copy()
    bg_val_all = bg_all[bg_all["split"] == "validation"].copy()
    bg_ho = bg_all[bg_all["split"] == "station_holdout"].copy()

    # Sample train
    summary = {}
    prov_dfs = {}

    info_tr, _ = build_split_v3(
        "train", bg_train, tr_vessels, args.n_train, 0.42, neg_std, pos_std, rng, args.out_dir, hms_prob=0.15
    )
    summary["train"] = info_tr
    prov_dfs["train"] = pd.read_csv(args.out_dir / "train_provenance.csv")
    print(f"Generated train: {info_tr['rows']} rows, zero={info_tr['zero_fraction']:.3f}, mean_y={info_tr['mean_share']:.3f}, strata={info_tr['strata_counts']}")

    # Partition validation run_ids disjointly (2 halves: one for val_hard_event, one for val_standard)
    bg_val_part = assign_balanced_run_folds(bg_val_all, n_folds=2)
    bg_val_hard = bg_val_part[bg_val_part["cv_fold"] == 0].copy()
    bg_val_std = bg_val_part[bg_val_part["cv_fold"] == 1].copy()

    info_vh, _ = build_split_v3(
        "val_hard_event", bg_val_hard, va_vessels, args.n_val_hard, 0.50, neg_hard, pos_hard, rng, args.out_dir, hms_prob=0.15
    )
    summary["val_hard_event"] = info_vh
    prov_dfs["val_hard_event"] = pd.read_csv(args.out_dir / "val_hard_event_provenance.csv")
    print(f"Generated val_hard_event: {info_vh['rows']} rows, zero={info_vh['zero_fraction']:.3f}, neg_strata={info_vh['neg_strata_counts']}")

    info_vs, _ = build_split_v3(
        "val_standard", bg_val_std, va_vessels, args.n_val_std, 0.42, neg_std, pos_std, rng, args.out_dir, hms_prob=0.15
    )
    summary["val_standard"] = info_vs
    prov_dfs["val_standard"] = pd.read_csv(args.out_dir / "val_standard_provenance.csv")
    print(f"Generated val_standard: {info_vs['rows']} rows, zero={info_vs['zero_fraction']:.3f}, strata={info_vs['strata_counts']}")

    info_ho, _ = build_split_v3(
        "holdout", bg_ho, ho_vessels, args.n_holdout, 0.42, neg_std, pos_std, rng, args.out_dir, hms_prob=0.15
    )
    summary["holdout"] = info_ho
    prov_dfs["holdout"] = pd.read_csv(args.out_dir / "holdout_provenance.csv")
    print(f"Generated holdout: {info_ho['rows']} rows, zero={info_ho['zero_fraction']:.3f}, strata={info_ho['strata_counts']}")

    # Cross-split leakage assertions (including disjoint bg_run_id between val_hard_event & val_standard!)
    tr_p = prov_dfs["train"]
    assert set(prov_dfs["val_hard_event"]["bg_run_id"]).isdisjoint(prov_dfs["val_standard"]["bg_run_id"])
    for vp_name in ("val_standard", "val_hard_event", "holdout"):
        vp = prov_dfs[vp_name]
        assert set(tr_p["bg_run_id"]).isdisjoint(vp["bg_run_id"]), f"bg_run_id overlap train vs {vp_name}"
        if vp_name != "holdout":
            assert set(prov_dfs["holdout"]["bg_run_id"]).isdisjoint(vp["bg_run_id"])

    def _nonempty_groups(df: pd.DataFrame) -> set[str]:
        s = df["vessel_group_id"].dropna().astype(str)
        return {x for x in s if x}

    assert _nonempty_groups(tr_p).isdisjoint(_nonempty_groups(prov_dfs["val_standard"]))
    assert _nonempty_groups(tr_p).isdisjoint(_nonempty_groups(prov_dfs["val_hard_event"]))
    assert _nonempty_groups(tr_p).isdisjoint(_nonempty_groups(prov_dfs["holdout"]))
    assert _nonempty_groups(prov_dfs["val_standard"]).isdisjoint(_nonempty_groups(prov_dfs["holdout"]))
    assert _nonempty_groups(prov_dfs["val_hard_event"]).isdisjoint(_nonempty_groups(prov_dfs["holdout"]))

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

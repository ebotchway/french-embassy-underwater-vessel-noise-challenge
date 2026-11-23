#!/usr/bin/env python3
"""Generate leak-free, calibration-randomized, strict-ship & sharp-block datasets (Modeling v4)."""
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
from modeling_v3_simulator import (  # noqa: E402
    REF_DB,
    assign_disjoint_vessel_sources_v3,
    db_to_lin_fixed_ref,
    read_windows_fast,
)

BG_MANIFEST_DEFAULT = ROOT / "reports/eda/background_split_manifest_v2.csv"
VTUAD_PARQUET_DEFAULT = ROOT / "data/processed/vtuad_decidecade_2s_minutes.parquet"
HMS_PARQUET_DEFAULT = ROOT / "data/processed/hearmyship_decidecade_2s_clips.parquet"
OUT_DIR_DEFAULT = ROOT / "data/synthetic/modeling_v4"

LOG10_F_NORM = np.log10(EXACT_CENTRES / 500.0)
HIGH_BAND_ATTEN_WEIGHT = np.maximum(0.0, np.log10(EXACT_CENTRES / 300.0))
NORM_BAND_COORD = np.linspace(0.0, 1.0, 25)

STRATA_ORDER = ("strict_hf_ship", "reef_active", "glitch_drop", "vlf_surf", "ambient_normal")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_backgrounds_v4(bg_df: pd.DataFrame) -> pd.DataFrame:
    """Profile all 15x25 SA background windows into 5 strict physical strata."""
    work = bg_df.copy()
    work["bg_bb_rng_db"] = 0.0
    work["bg_hf_rng_db"] = 0.0
    work["bg_mf_rng_db"] = 0.0
    work["bg_w6_hf"] = 15
    work["bg_argmax_hf"] = 0
    work["bg_q_d63"] = 0.0
    work["bg_drop_db"] = 0.0
    work["stratum"] = "ambient_normal"

    for pq_path, grp in work.groupby("source_parquet", sort=True):
        arr = pd.read_parquet(str(pq_path), columns=BANDS).to_numpy(dtype=np.float64)
        idx_arr = grp.index.to_numpy()
        pos_mat = np.asarray([json.loads(s) for s in grp["row_positions_json"]], dtype=np.int64)
        W = arr[pos_mat]
        W_med = np.median(W, axis=(1, 2), keepdims=True)
        E = np.power(10.0, np.clip(W - W_med, -60.0, 60.0) / 10.0)

        bb_db = 10.0 * np.log10(np.maximum(E.sum(axis=2), 1e-30))
        mf_db = 10.0 * np.log10(np.maximum(E[:, :, 11:17].sum(axis=2), 1e-30))
        hf_db = 10.0 * np.log10(np.maximum(E[:, :, 17:25].sum(axis=2), 1e-30))

        bb_rng = bb_db.max(axis=1) - bb_db.min(axis=1)
        mf_rng = mf_db.max(axis=1) - mf_db.min(axis=1)
        hf_rng = hf_db.max(axis=1) - hf_db.min(axis=1)
        amx_hf = np.argmax(hf_db, axis=1)
        w6_hf = (hf_db >= hf_db.max(axis=1, keepdims=True) - 6.0).sum(axis=1)

        argmin_bb = np.argmin(bb_db, axis=1)
        q_rows = W[np.arange(len(W)), argmin_bb, :]
        q_d63 = q_rows[:, 23] - q_rows[:, 21]
        drop_db = np.median(bb_db, axis=1) - bb_db.min(axis=1)

        is_glitch = drop_db > 14.0
        is_ship = (~is_glitch) & (hf_rng >= 10.0) & (mf_rng >= 8.0) & (amx_hf >= 1) & (amx_hf <= 13) & (w6_hf <= 8)
        is_reef = (~is_glitch) & (~is_ship) & (q_d63 > 1.5) & (bb_rng >= 8.0)
        is_vlf = (~is_glitch) & (~is_ship) & (~is_reef) & (bb_rng >= 10.0)

        stratum = np.full(len(W), "ambient_normal", dtype=object)
        stratum[is_vlf] = "vlf_surf"
        stratum[is_reef] = "reef_active"
        stratum[is_ship] = "strict_hf_ship"
        stratum[is_glitch] = "glitch_drop"

        work.loc[idx_arr, "bg_bb_rng_db"] = bb_rng
        work.loc[idx_arr, "bg_hf_rng_db"] = hf_rng
        work.loc[idx_arr, "bg_mf_rng_db"] = mf_rng
        work.loc[idx_arr, "bg_w6_hf"] = w6_hf
        work.loc[idx_arr, "bg_argmax_hf"] = amx_hf
        work.loc[idx_arr, "bg_q_d63"] = q_d63
        work.loc[idx_arr, "bg_drop_db"] = drop_db
        work.loc[idx_arr, "stratum"] = stratum

    return work


def apply_channel_warp_v4(V: np.ndarray, active_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply wide propagation tilt, modal ripple, and cell jitter to vessel energy."""
    out = V.copy()
    if not active_mask.any():
        return out
    tilt_db_per_dec = float(np.clip(rng.normal(0.0, 3.5), -8.0, 8.0))
    ripple_amp = float(rng.uniform(0.0, 2.5))
    ripple_freq = float(rng.uniform(0.5, 2.0))
    ripple_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    spectral_warp_db = (
        tilt_db_per_dec * LOG10_F_NORM
        + ripple_amp * np.cos(2.0 * np.pi * ripple_freq * NORM_BAND_COORD + ripple_phase)
    )
    jitter_db = rng.normal(0.0, 0.70, size=(15, 25))
    factor = np.power(10.0, (spectral_warp_db[None, :] + jitter_db) / 10.0)
    out[active_mask] *= factor[active_mask]
    return np.maximum(out, 0.0)


def synthesize_vessel_trajectory_v4(
    source_matrix: np.ndarray,
    rel_power: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, int]:
    """Generate physical 15x25 vessel trajectory favoring sharp 2-5 min blocks & CPA bells."""
    n_avail = source_matrix.shape[0]
    t_idx = np.arange(15, dtype=np.float64)
    u = float(rng.random())

    # 40%: short_block_vtuad (2-5 interior consecutive minutes matching real short events)
    if u < 0.40:
        dur = int(rng.integers(2, 6))
        if n_avail >= dur:
            src_start = int(rng.integers(0, n_avail - dur + 1))
            block = source_matrix[src_start : src_start + dur] * rel_power[src_start : src_start + dur, None]
        elif n_avail == 1:
            block = np.repeat(source_matrix, dur, axis=0)
        else:
            idxs = np.clip(np.round(np.linspace(0, n_avail - 1, dur)).astype(int), 0, n_avail - 1)
            block = source_matrix[idxs] * rel_power[idxs, None]

        max_start = max(1, 14 - dur)
        ins_start = int(rng.integers(1, max_start + 1))
        V = np.zeros(SHAPE, dtype=np.float64)
        V[ins_start : ins_start + dur] = block
        # 40% mild edge taper (0.25..0.55) on boundary minutes of short block; 60% sharp step onset
        if rng.random() < 0.40 and dur >= 3:
            V[ins_start] *= float(rng.uniform(0.25, 0.65))
            V[ins_start + dur - 1] *= float(rng.uniform(0.25, 0.65))
        V = apply_channel_warp_v4(V, active_mask=(V.sum(axis=1) > 0), rng=rng)
        return V, "short_block_vtuad", dur

    # Prepare 15-row spectral template for continuous CPA modes
    if n_avail == 1:
        base = np.repeat(source_matrix, 15, axis=0)
    else:
        start = int(rng.integers(0, n_avail))
        step_dir = 1 if rng.random() < 0.5 else -1
        idxs = np.clip(start + step_dir * np.arange(15), 0, n_avail - 1)
        base = source_matrix[idxs].copy()
    row_tot = np.maximum(base.sum(axis=1, keepdims=True), 1e-30)
    base_shape = base / row_tot * float(row_tot.mean())

    if u < 0.80:
        mode = "cpa_sharp_bell"
        t_cpa = float(rng.uniform(1.5, 12.5))
        width = float(rng.uniform(0.55, 2.2))
        exponent = float(rng.uniform(0.95, 1.55))
        dist_rel = np.sqrt(1.0 + ((t_idx - t_cpa) / width) ** 2)
        env = np.power(dist_rel, -2.0 * exponent)
        hf_atten_db = -float(rng.uniform(0.8, 3.5)) * (dist_rel[:, None] - 1.0) * HIGH_BAND_ATTEN_WEIGHT[None, :]
        V = base_shape * (env[:, None] * np.power(10.0, hf_atten_db / 10.0))
        dur = int((env > 0.1 * env.max()).sum())
    elif u < 0.92:
        mode = "cpa_asym_or_double"
        if rng.random() < 0.65:
            t_cpa = float(rng.uniform(2.0, 12.0))
            widths = np.where(t_idx <= t_cpa, float(rng.uniform(0.6, 3.2)), float(rng.uniform(0.6, 3.2)))
            exponent = float(rng.uniform(0.95, 1.45))
            dist_rel = np.sqrt(1.0 + ((t_idx - t_cpa) / widths) ** 2)
            env = np.power(dist_rel, -2.0 * exponent)
        else:
            t1 = float(rng.uniform(2.0, 5.5))
            t2 = float(rng.uniform(8.5, 12.5))
            w1, w2 = float(rng.uniform(0.6, 1.6)), float(rng.uniform(0.6, 1.6))
            d1 = np.sqrt(1.0 + ((t_idx - t1) / w1) ** 2)
            d2 = np.sqrt(1.0 + ((t_idx - t2) / w2) ** 2)
            env = 0.6 * np.power(d1, -2.2) + np.power(d2, -2.2)
            dist_rel = np.minimum(d1, d2)
        hf_atten_db = -float(rng.uniform(0.8, 3.2)) * (dist_rel[:, None] - 1.0) * HIGH_BAND_ATTEN_WEIGHT[None, :]
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

    V = apply_channel_warp_v4(V, active_mask=(V.sum(axis=1) > 0), rng=rng)
    return V, mode, dur


def apply_hydrophone_calibration(
    B: np.ndarray,
    V: np.ndarray,
    rng: np.random.Generator,
    prob: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Multiply B and V identically by a smooth hydrophone transfer gain G(f) = 10^(h_db/10)."""
    if rng.random() >= prob:
        return B, V, False
    tilt = float(rng.uniform(-4.0, 9.0))
    amp = float(rng.uniform(0.0, 2.5))
    freq = float(rng.uniform(0.5, 1.8))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    h_db = tilt * LOG10_F_NORM + amp * np.cos(2.0 * np.pi * freq * NORM_BAND_COORD + phase)
    G = np.power(10.0, h_db / 10.0)[None, :]
    return B * G, V * G, True


def sample_requested_share_v4(rng: np.random.Generator, zero_prob: float) -> float:
    """Sample target share across exact 0.0 and calibrated continuous vessel share regimes."""
    u = float(rng.random())
    if u < zero_prob:
        return 0.0
    rem = (u - zero_prob) / max(1.0 - zero_prob, 1e-9)
    if rem < 0.30:
        return float(rng.uniform(0.005, 0.08))
    if rem < 0.70:
        return float(rng.uniform(0.08, 0.30))
    if rem < 0.93:
        return float(rng.uniform(0.30, 0.68))
    return float(rng.uniform(0.68, 0.92))


def sample_minute_mask_v4(rng: np.random.Generator, stratum: str) -> np.ndarray:
    """Generate whole-minute mask matching Test.csv missing-minute telemetry."""
    obs = np.ones(15, dtype=bool)
    p_m01 = 0.60 if stratum == "reef_active" else 0.30
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


def _build_stratum_queues(pool_df: pd.DataFrame, rng: np.random.Generator) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    stratum_queues: dict[str, list[int]] = {}
    for st in ("strict_hf_ship", "reef_active", "glitch_drop", "vlf_surf"):
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


def _pop_from_stratum(
    stratum_queues: dict[str, list[int]],
    desired_stratum: str,
    ambient_station_queues: dict[str, list[int]],
    used_set: set[int],
) -> int:
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

    order = [desired_stratum] + [s for s in STRATA_ORDER if s != desired_stratum]
    for st in order:
        if st == "ambient_normal":
            idx = _pop_amb()
            if idx is not None:
                return idx
        else:
            q = stratum_queues.get(st, [])
            while q:
                idx = q.pop()
                if idx not in used_set:
                    used_set.add(idx)
                    return idx
    raise RuntimeError("Exhausted all background windows in pool")


def assign_balanced_run_folds(bg_sub: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Partition background run_ids into n_folds disjoint folds balancing rare strata & size."""
    out = bg_sub.copy()
    run_info = []
    for rid, grp in out.groupby("run_id", sort=True):
        vc = grp["stratum"].value_counts()
        run_info.append(
            {
                "run_id": str(rid),
                "n": len(grp),
                "glitch": int(vc.get("glitch_drop", 0)),
                "reef": int(vc.get("reef_active", 0)),
                "ship": int(vc.get("strict_hf_ship", 0)),
                "vlf": int(vc.get("vlf_surf", 0)),
            }
        )
    rdf = pd.DataFrame(run_info).sort_values(["glitch", "reef", "ship", "vlf", "n"], ascending=False)
    fold_stats = [{"n": 0, "glitch": 0, "reef": 0, "ship": 0, "vlf": 0} for _ in range(n_folds)]
    run_to_fold: dict[str, int] = {}
    for row in rdf.itertuples(index=False):
        if row.glitch > 0:
            f = min(range(n_folds), key=lambda k: (fold_stats[k]["glitch"], fold_stats[k]["n"]))
        elif row.reef > 0:
            f = min(range(n_folds), key=lambda k: (fold_stats[k]["reef"], fold_stats[k]["n"]))
        elif row.ship > 0:
            f = min(range(n_folds), key=lambda k: (fold_stats[k]["ship"], fold_stats[k]["n"]))
        elif row.vlf > 0:
            f = min(range(n_folds), key=lambda k: (fold_stats[k]["vlf"], fold_stats[k]["n"]))
        else:
            f = min(range(n_folds), key=lambda k: fold_stats[k]["n"])
        run_to_fold[row.run_id] = f
        fold_stats[f]["n"] += int(row.n)
        fold_stats[f]["glitch"] += int(row.glitch)
        fold_stats[f]["reef"] += int(row.reef)
        fold_stats[f]["ship"] += int(row.ship)
        fold_stats[f]["vlf"] += int(row.vlf)
    out["cv_fold"] = out["run_id"].astype(str).map(run_to_fold)
    return out


def _pick_stratum_from_probs(probs: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=np.float64)
    p /= p.sum()
    return str(keys[int(rng.choice(len(keys), p=p))])


def build_split_v4(
    split_name: str,
    bg_pool: pd.DataFrame,
    vessel_pool: dict,
    n_target: int,
    zero_prob: float,
    cal_prob: float,
    neg_strata_probs: dict[str, float],
    pos_strata_probs: dict[str, float],
    rng: np.random.Generator,
    out_dir: Path,
    hms_prob: float = 0.12,
) -> dict:
    """Generate and persist NPZ matrices + provenance CSV for one Modeling v4 split."""
    n = min(n_target, len(bg_pool))
    chosen_records: list[tuple[int, int, float]] = []
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
                q = sample_requested_share_v4(rng, zero_prob=zero_prob)
                st = _pick_stratum_from_probs(neg_strata_probs if q == 0.0 else pos_strata_probs, rng)
                bg_idx = _pop_from_stratum(sq, st, aq, used_set)
                chosen_records.append((bg_idx, f, q))
        rng.shuffle(chosen_records)
    else:
        sq, aq = _build_stratum_queues(bg_pool, rng)
        for _ in range(n):
            q = sample_requested_share_v4(rng, zero_prob=zero_prob)
            st = _pick_stratum_from_probs(neg_strata_probs if q == 0.0 else pos_strata_probs, rng)
            bg_idx = _pop_from_stratum(sq, st, aq, used_set)
            chosen_records.append((bg_idx, -1, q))
        rng.shuffle(chosen_records)

    bg_indices = [r[0] for r in chosen_records]
    assert len(set(bg_indices)) == len(bg_indices)
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
            mode, dur = "background_only", 0
            v_dataset, v_source_id, v_group_id, v_class = "", "", "", ""
            alpha = 0.0
        else:
            src = _sample_vessel_from_pool(vessel_pool, fold_arg, hms_prob=hms_prob, rng=rng)
            V_raw, mode, dur = synthesize_vessel_trajectory_v4(src["matrix"], src["rel_power"], rng)
            v_sum = float(V_raw.sum())
            if v_sum <= 0.0:
                V = np.zeros(SHAPE, dtype=np.float64)
                alpha = 0.0
            else:
                b_sum = float(B.sum())
                alpha = (q * b_sum) / ((1.0 - q) * v_sum)
                V = alpha * V_raw
            v_dataset = src["dataset"]
            v_source_id = src["source_id"]
            v_group_id = src["group_id"]
            v_class = src["vessel_class"]

        B_cal, V_cal, cal_applied = apply_hydrophone_calibration(B, V, rng, prob=cal_prob)
        M = B_cal + V_cal

        obs_min_mask = sample_minute_mask_v4(rng, str(rec.stratum))
        obs_cell_mask = np.repeat(obs_min_mask[:, None], 25, axis=1)

        m_obs_sum = float(M[obs_cell_mask].sum())
        v_obs_sum = float(V_cal[obs_cell_mask].sum())
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
                "cal_warped": bool(cal_applied),
                "bg_run_id": str(rec.run_id),
                "vessel_dataset": v_dataset,
                "vessel_source_id": v_source_id,
                "vessel_group_id": v_group_id,
                "vessel_class": v_class,
                "trajectory_mode": mode,
                "vessel_duration_min": dur,
                "requested_share": q,
                "observed_share": y_obs,
                "n_obs_minutes": int(obs_min_mask.sum()),
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

    return {
        "split": split_name,
        "rows": n,
        "zero_fraction": float((y == 0.0).mean()),
        "mean_share": float(y.mean()),
        "cal_warped_fraction": float(prov_df["cal_warped"].mean()),
        "strata_counts": {k: int(v) for k, v in prov_df["bg_stratum"].value_counts().items()},
        "neg_strata_counts": {k: int(v) for k, v in prov_df.loc[prov_df["observed_share"] == 0, "bg_stratum"].value_counts().items()},
        "npz_sha256": _sha256(npz_path),
    }


def run_self_test() -> None:
    """Verify h(f) calibration linearity, +37.5 dB shift consistency, and v4 trajectory synthesis."""
    rng = np.random.default_rng(7)
    B_db = rng.uniform(75.0, 110.0, size=(15, 25))
    B = db_to_lin_fixed_ref(B_db)
    V_raw = rng.uniform(0.1, 2.0, size=(15, 25))
    q = 0.25
    alpha = (q * B.sum()) / ((1.0 - q) * V_raw.sum())
    V = alpha * V_raw
    B_cal, V_cal, applied = apply_hydrophone_calibration(B, V, rng, prob=1.0)
    assert applied
    # Multiplying both B and V by G(f) preserves V(t,f)/B(t,f) identically at every (t,f) cell!
    assert np.allclose(V_cal / B_cal, V / B, atol=1e-12)
    M_cal = B_cal + V_cal
    y_cal = float(V_cal.sum() / M_cal.sum())
    assert 0.0 < y_cal < 1.0

    # Global +37.5 dB shift check
    B_sh = db_to_lin_fixed_ref(B_db + 37.5)
    alpha_s = (q * B_sh.sum()) / ((1.0 - q) * V_raw.sum())
    M_sh = B_sh + alpha_s * V_raw
    assert np.allclose(10.0 * np.log10(M_sh / (B + V)), 37.5, atol=1e-10)

    src = rng.uniform(0.01, 1.0, size=(6, 25))
    src /= src.sum(axis=1, keepdims=True)
    rp = rng.uniform(0.5, 2.0, size=6)
    modes = set()
    for _ in range(60):
        V_syn, md, _ = synthesize_vessel_trajectory_v4(src, rp, rng)
        modes.add(md)
        assert V_syn.shape == SHAPE and np.isfinite(V_syn).all() and (V_syn >= 0.0).all()
    assert len(modes) >= 3
    print("modeling_v4_simulator self-test: PASS")


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
    print("Profiling South African background windows into 5 strict physical strata...", flush=True)
    bg_all = profile_backgrounds_v4(pd.read_csv(args.bg_manifest))
    print(pd.crosstab(bg_all["split"], bg_all["stratum"], margins=True).to_string())

    vt_df = pd.read_parquet(args.vtuad_parquet)
    hms_df = pd.read_parquet(args.hms_parquet) if args.hms_parquet.is_file() else pd.DataFrame(columns=["source_split"])
    tr_vessels, va_vessels, ho_vessels = assign_disjoint_vessel_sources_v3(vt_df, hms_df, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    neg_std = {"strict_hf_ship": 0.45, "reef_active": 0.22, "vlf_surf": 0.10, "glitch_drop": 0.06, "ambient_normal": 0.17}
    pos_std = {"strict_hf_ship": 0.25, "reef_active": 0.22, "vlf_surf": 0.10, "glitch_drop": 0.04, "ambient_normal": 0.39}
    neg_hard = {"strict_hf_ship": 0.60, "reef_active": 0.30, "glitch_drop": 0.10, "vlf_surf": 0.00, "ambient_normal": 0.00}
    pos_hard = {"strict_hf_ship": 0.35, "reef_active": 0.30, "glitch_drop": 0.05, "vlf_surf": 0.00, "ambient_normal": 0.30}

    bg_train = bg_all[bg_all["split"] == "train"].copy()
    bg_val_part = assign_balanced_run_folds(bg_all[bg_all["split"] == "validation"].copy(), n_folds=2)
    bg_val_hard = bg_val_part[bg_val_part["cv_fold"] == 0].copy()
    bg_val_std = bg_val_part[bg_val_part["cv_fold"] == 1].copy()
    bg_ho = bg_all[bg_all["split"] == "station_holdout"].copy()

    summary = {}
    prov_dfs = {}

    for sp_name, pool, vp, nt, zp, cp, ns, ps in (
        ("train", bg_train, tr_vessels, args.n_train, 0.45, 0.65, neg_std, pos_std),
        ("val_cal_hard", bg_val_hard, va_vessels, args.n_val_hard, 0.50, 1.00, neg_hard, pos_hard),
        ("val_standard", bg_val_std, va_vessels, args.n_val_std, 0.45, 0.65, neg_std, pos_std),
        ("holdout", bg_ho, ho_vessels, args.n_holdout, 0.45, 0.00, neg_std, pos_std),
    ):
        info = build_split_v4(sp_name, pool, vp, nt, zp, cp, ns, ps, rng, args.out_dir, hms_prob=0.12)
        summary[sp_name] = info
        prov_dfs[sp_name] = pd.read_csv(args.out_dir / f"{sp_name}_provenance.csv")
        print(f"Generated {sp_name}: rows={info['rows']}, zero={info['zero_fraction']:.3f}, "
              f"mean_y={info['mean_share']:.3f}, cal={info['cal_warped_fraction']:.2f}, neg_strata={info['neg_strata_counts']}")

    # Strict cross-split & cross-fold leakage assertions
    tr_p = prov_dfs["train"]
    assert set(prov_dfs["val_cal_hard"]["bg_run_id"]).isdisjoint(prov_dfs["val_standard"]["bg_run_id"])
    for vp_name in ("val_standard", "val_cal_hard", "holdout"):
        vp = prov_dfs[vp_name]
        assert set(tr_p["bg_run_id"]).isdisjoint(vp["bg_run_id"]), f"bg_run_id overlap train vs {vp_name}"
        if vp_name != "holdout":
            assert set(prov_dfs["holdout"]["bg_run_id"]).isdisjoint(vp["bg_run_id"])

    def _nonempty_groups(df: pd.DataFrame) -> set[str]:
        s = df["vessel_group_id"].dropna().astype(str)
        return {x for x in s if x}

    for vp_name in ("val_standard", "val_cal_hard", "holdout"):
        assert _nonempty_groups(tr_p).isdisjoint(_nonempty_groups(prov_dfs[vp_name]))
        if vp_name != "holdout":
            assert _nonempty_groups(prov_dfs[vp_name]).isdisjoint(_nonempty_groups(prov_dfs["holdout"]))

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

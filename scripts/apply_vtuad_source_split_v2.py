from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

features_path = Path(
    "reports/eda/vtuad_minute_band_power_v1.csv"
)
split_path = Path(
    "reports/eda/vtuad_source_split_v2.csv"
)
output = Path(
    "reports/eda/vtuad_minute_band_power_split_v2.csv"
)

features = pd.read_csv(
    features_path, dtype={"hull_id": "string"}
)
sources = pd.read_csv(
    split_path, dtype={"hull_id": "string"}
)

assert len(features) == 1892
assert len(sources) == 525
assert sources["clip_uid"].is_unique

bands = [c for c in features if c.startswith("TOL_")]
assert len(bands) == 25

mapping = sources[
    ["clip_uid", "hull_id", "label", "source_split"]
].rename(columns={
    "hull_id": "new_hull_id",
    "label": "new_label",
    "source_split": "new_split"
})

updated = features.merge(
    mapping,
    on="clip_uid",
    how="left",
    validate="many_to_one"
)

assert updated["new_split"].notna().all()
assert updated["hull_id"].eq(updated["new_hull_id"]).all()
assert updated["label"].eq(updated["new_label"]).all()

updated["source_split"] = updated["new_split"]

updated = updated.drop(columns=[
    "new_hull_id", "new_label", "new_split"
])

assert updated["source_split"].isin(
    ["train", "validation"]
).all()

train = updated[updated["source_split"] == "train"]
val = updated[updated["source_split"] == "validation"]

assert set(train["hull_id"]).isdisjoint(
    set(val["hull_id"])
)
assert updated[bands].notna().all().all()
assert np.isfinite(updated[bands].to_numpy()).all()
assert (updated[bands] >= 0).all().all()

assert updated["clip_uid"].nunique() == 487
assert len(updated) == len(features)

# No acoustic measurements should change during relabelling.
assert np.array_equal(
    updated[bands].to_numpy(),
    features[bands].to_numpy()
)

assert np.array_equal(
    updated["relative_total_power"].to_numpy(),
    features["relative_total_power"].to_numpy()
)

output.parent.mkdir(parents=True, exist_ok=True)
updated.to_csv(output, index=False)

print("=== VTUAD SPLIT V2 FEATURE AUDIT ===")
print("Minute records:", len(updated))

print("\nMinutes by split:")
print(updated["source_split"].value_counts().to_string())

print("\nClips by split:")
print(
    updated.groupby("source_split")["clip_uid"]
    .nunique().to_string()
)

print("\nHulls by split:")
print(
    updated.groupby("source_split")["hull_id"]
    .nunique().to_string()
)

print("\nMinutes by vessel class:")
print(
    pd.crosstab(
        updated["label"], updated["source_split"]
    ).to_string()
)

print("\nHull overlap: 0")
print("Output SHA256:",
      hashlib.sha256(output.read_bytes()).hexdigest())
print("Saved:", output)
print("SPLIT V2 FEATURE VALIDATION: PASS")

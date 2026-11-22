#!/usr/bin/env python3
"""Create a class-aware, hull-isolated VTUAD source split."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

INPUT = Path("reports/eda/vtuad_source_split_v1.csv")
OUTPUT = Path("reports/eda/vtuad_source_split_v2.csv")

SEED = 42
VALIDATION_FRACTION = 0.20

df = pd.read_csv(INPUT, dtype={"hull_id": "string"})

assert len(df) == 525
assert df["clip_uid"].is_unique
assert df["path"].is_unique

eligible = df[df["label"].ne("background")].copy()

assert len(eligible) == 487
assert eligible["hull_id"].notna().all()
assert eligible["hull_id"].nunique() == 172

# Confirm that each hull has exactly one class.
hull_label_counts = eligible.groupby("hull_id")["label"].nunique()
assert hull_label_counts.max() == 1

# One row per hull, with its class.
hulls = (
    eligible[["hull_id", "label"]]
    .drop_duplicates()
    .sort_values(["label", "hull_id"])
)

rng = np.random.default_rng(SEED)

validation_hulls = set()
single_hull_classes = []

for label, group in hulls.groupby("label", sort=True):
    identities = group["hull_id"].tolist()
    n_hulls = len(identities)

    if n_hulls == 1:
        single_hull_classes.append(label)
        continue

    n_validation = max(
        1,
        round(n_hulls * VALIDATION_FRACTION),
    )
    n_validation = min(n_validation, n_hulls - 1)

    chosen = rng.choice(
        identities,
        size=n_validation,
        replace=False,
    )

    validation_hulls.update(chosen.tolist())

df["source_split"] = "background_reference"

df.loc[
    df["label"].ne("background"),
    "source_split",
] = "train"

df.loc[
    df["hull_id"].isin(validation_hulls)
    & df["label"].ne("background"),
    "source_split",
] = "validation"

train = df[df["source_split"].eq("train")]
validation = df[df["source_split"].eq("validation")]
background = df[df["source_split"].eq("background_reference")]

train_hulls = set(train["hull_id"])
validation_hulls_actual = set(validation["hull_id"])

assert train_hulls.isdisjoint(validation_hulls_actual)
assert len(train) + len(validation) == 487
assert len(background) == 38
assert len(df) == 525

assert train["hull_id"].notna().all()
assert validation["hull_id"].notna().all()
assert not train["label"].eq("background").any()
assert not validation["label"].eq("background").any()

# Every class must appear in training.
all_classes = set(eligible["label"])
assert set(train["label"]) == all_classes

# Every class with at least two hulls must appear in validation.
multi_hull_classes = set(
    hulls.groupby("label")["hull_id"]
    .nunique()
    .loc[lambda counts: counts >= 2]
    .index
)
assert set(validation["label"]) == multi_hull_classes

# Each hull belongs to exactly one active partition.
active = df[df["source_split"].isin(["train", "validation"])]
assert active.groupby("hull_id")["source_split"].nunique().max() == 1

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=== VTUAD CLASS-AWARE SOURCE SPLIT V2 ===")

print("\nClip counts:")
print(df["source_split"].value_counts().to_string())

print("\nUnique hull counts:")
print(
    active.groupby("source_split")["hull_id"]
    .nunique().to_string()
)

print("\nClips by class:")
print(
    pd.crosstab(
        active["label"],
        active["source_split"],
    ).to_string()
)

print("\nHulls by class:")
print(
    pd.crosstab(
        hulls["label"],
        np.where(
            hulls["hull_id"].isin(validation_hulls_actual),
            "validation",
            "train",
        ),
    ).to_string()
)

print("\nClasses with only one hull:")
print(", ".join(single_hull_classes))

print("\nHull overlap:",
      len(train_hulls & validation_hulls_actual))

digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
print("Manifest SHA256:", digest)

print("\nSaved:", OUTPUT)
print("CLASS-AWARE HULL SPLIT: PASS")

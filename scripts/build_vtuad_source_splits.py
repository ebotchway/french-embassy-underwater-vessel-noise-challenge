from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

INPUT = Path("reports/eda/vtuad_metadata_reconciliation.csv")
OUTPUT = Path("reports/eda/vtuad_source_split_v1.csv")
SEED = 42

df = pd.read_csv(INPUT, dtype={"hull_id": "string"})

required = {
    "clip_uid", "path", "label", "hull_id",
    "directory_category", "exists",
    "wav_duration"
}
assert required.issubset(df.columns)
assert len(df) == 525
assert df["clip_uid"].is_unique
assert df["path"].is_unique
assert df["exists"].all()

# Background labels must not be used as vessel additions.
background = df["label"].eq("background")

# Candidate vessel recordings have a non-background class label.
vessel = ~background

assert int(background.sum()) == 38
assert int(vessel.sum()) == 487

# Explicitly exclude missing vessel identities.
known = vessel & df["hull_id"].notna()
unknown = vessel & df["hull_id"].isna()

df["source_split"] = "excluded"

df.loc[background, "source_split"] = "background_reference"
df.loc[unknown, "source_split"] = "unknown_identity"

eligible = df.loc[known].copy()
eligible = eligible.reset_index()

assert len(eligible) > 0
assert eligible["hull_id"].nunique() >= 2
assert eligible["wav_duration"].gt(0).all()

# Split complete hull groups, never individual clips.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_idx, val_idx = next(
    splitter.split(
        eligible,
        groups=eligible["hull_id"]
    )
)

train_rows = eligible.iloc[train_idx]
val_rows = eligible.iloc[val_idx]

df.loc[train_rows["index"], "source_split"] = "train"
df.loc[val_rows["index"], "source_split"] = "validation"

# Verify that vessel identities do not cross the partitions.
train_hulls = set(train_rows["hull_id"])
val_hulls = set(val_rows["hull_id"])

assert train_hulls.isdisjoint(val_hulls)

active = df[df["source_split"].isin(["train", "validation"])]

assert len(active) == len(eligible)
assert active["hull_id"].notna().all()
assert active["wav_duration"].gt(0).all()
assert not active["label"].eq("background").any()

# Every eligible clip has exactly one partition.
assert (
    df.groupby("clip_uid")["source_split"]
    .nunique()
    .max() == 1
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=== VTUAD SOURCE SPLIT V1 ===")
print(df["source_split"].value_counts().to_string())

print("\nDistinct hulls")
print("Train:", len(train_hulls))
print("Validation:", len(val_hulls))
print("Overlapping:", len(train_hulls & val_hulls))

print("\nClass distribution:")
print(
    pd.crosstab(
        df["label"], df["source_split"]
    ).to_string()
)

print("\nSaved:", OUTPUT)
print("HULL-LEVEL SPLIT VALIDATION: PASS")

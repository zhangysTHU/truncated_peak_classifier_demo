"""Load peak features by sequencing-sample/peak identity, never by row position."""

from pathlib import Path
import numpy as np
import pandas as pd


def load_labeled_samples(sample_dir, label_dir, expected_features=None):
    sample_dir, label_dir = Path(sample_dir), Path(label_dir)
    sample_files = sorted(sample_dir.glob("*_sample_prepared.csv"))
    if not sample_files:
        raise FileNotFoundError(f"No prepared samples in {sample_dir}")
    sample_ids = {p.name.removesuffix("_sample_prepared.csv") for p in sample_files}
    label_ids = {p.name.removesuffix("_labels.csv") for p in label_dir.glob("*_labels.csv")}
    if sample_ids != label_ids:
        raise ValueError("Sample and label file IDs do not match")
    frames, feature_columns = [], None
    for sample_file in sample_files:
        sample_id = sample_file.name.removesuffix("_sample_prepared.csv")
        features = pd.read_csv(sample_file)
        labels = pd.read_csv(label_dir / f"{sample_id}_labels.csv")
        if "sample" not in features or not {"sample", "label"}.issubset(labels):
            raise ValueError(f"Missing sample/label columns for {sample_id}")
        columns = features.columns.drop("sample").tolist()
        if feature_columns is None:
            feature_columns = columns
        if columns != feature_columns:
            raise ValueError(f"Feature-column order differs for {sample_id}")
        if expected_features is not None and len(columns) != expected_features:
            raise ValueError(f"{sample_id}: expected {expected_features} features, found {len(columns)}. "
                             "Restore the actual full-feature input; zero-padding does not reproduce it.")
        if features["sample"].isna().any() or labels["sample"].isna().any():
            raise ValueError(f"Missing peak IDs in {sample_id}")
        joined = features.merge(labels[["sample", "label"]], on="sample", how="outer",
                                validate="one_to_one", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"Unmatched feature/label peak IDs in {sample_id}")
        joined = joined.drop(columns="_merge")
        if not joined["label"].isin([0, 1]).all():
            raise ValueError(f"Non-binary labels in {sample_id}")
        if not np.isfinite(joined[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid features in {sample_id}")
        joined.insert(0, "source_sample", sample_id)
        frames.append(joined)
    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(data)} peaks, {len(sample_files)} sequencing samples, "
          f"{len(feature_columns)} features; labels={data['label'].value_counts().to_dict()}")
    return data, feature_columns

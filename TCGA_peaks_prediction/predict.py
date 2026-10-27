"""Predict from an explicitly supplied checkpoint and channel-major feature CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def validate_input(path, channels=16):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if len(header) != 1 + channels * 40:
            raise ValueError(f"{path.name}: expected sample + {channels * 40} features; "
                             f"found {max(len(header) - 1, 0)} features. "
                             "The bundled 520-feature demo cannot reproduce a 640-feature model.")
        if header[0] != "sample" or len(set(header)) != len(header):
            raise ValueError(f"{path.name}: expected a unique header beginning with 'sample'.")
        count = 0
        for count, row in enumerate(reader, 1):
            if len(row) != len(header) or not row[0]:
                raise ValueError(f"{path.name}: malformed row {count + 1}.")
            try:
                valid = all(math.isfinite(float(x)) for x in row[1:])
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"{path.name}: non-finite/non-numeric feature in row {count + 1}.")
        if count == 0:
            raise ValueError(f"{path.name}: empty input.")
    return header, count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold-space", choices=["logit", "probability"])
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.channels <= 0 or args.batch_size <= 0:
        parser.error("channels and batch size must be positive")
    if not args.input_dir.is_dir():
        parser.error("--input-dir does not exist")
    files = sorted(args.input_dir.glob("*.csv"))
    if not files:
        parser.error("No CSV input files found")
    if not args.validate_only:
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error("Supply an existing --checkpoint; the historical AUROC 0.860 checkpoint "
                         "is not bundled and exploratory checkpoints are not substitutes.")
        if args.output_dir is None or args.threshold_space is None or args.threshold is None:
            parser.error("Prediction requires --output-dir, --threshold-space and --threshold. "
                         "The historical script used logit > 0.5; probability 0.5 is a different rule.")
        if not math.isfinite(args.threshold):
            parser.error("--threshold must be finite")
        if args.threshold_space == "probability" and not 0 <= args.threshold <= 1:
            parser.error("Probability threshold must be in [0, 1]")
    try:
        schemas = [validate_input(path, args.channels) for path in files]
    except ValueError as error:
        parser.error(str(error))
    if any(header != schemas[0][0] for header, _ in schemas):
        parser.error("All input files must have the same ordered feature columns")
    if args.validate_only:
        print(f"Validated {len(files)} files; {sum(n for _, n in schemas)} rows; "
              f"{args.channels * 40} features.")
        return
    destinations = [args.output_dir / path.name for path in files]
    if not args.overwrite and any(path.exists() for path in destinations):
        parser.error("Output already exists; choose a new directory or explicitly use --overwrite")

    import numpy as np
    import pandas as pd
    import torch
    from cross_attention import CrossAttentionModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    if "feature_columns" in checkpoint and checkpoint["feature_columns"] != schemas[0][0][1:]:
        parser.error("Checkpoint feature-column order does not match the input")
    if "channels" in checkpoint and checkpoint["channels"] != args.channels:
        parser.error("Checkpoint channel count does not match --channels")
    model = CrossAttentionModel(input_dim=args.channels, seq_len=40, hidden_dim=64, dropout_rate=0.15)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        parser.error(f"Checkpoint is incompatible with the row-column architecture: {error}")
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, destination in zip(files, destinations):
        data = pd.read_csv(path)
        features = data.iloc[:, 1:].to_numpy(dtype=np.float32)
        logits = []
        with torch.no_grad():
            for start in range(0, len(features), args.batch_size):
                batch = features[start:start + args.batch_size]
                inputs = torch.from_numpy(batch.reshape(-1, args.channels, 40).transpose(0, 2, 1).copy())
                logits.append(model(inputs).reshape(-1))
        logits = torch.cat(logits)
        if not torch.isfinite(logits).all():
            raise ValueError(f"Non-finite model output for {path.name}")
        probabilities = torch.sigmoid(logits)
        scores = logits if args.threshold_space == "logit" else probabilities
        pd.DataFrame({
            "sample": data["sample"], "prediction": logits.numpy(),
            "probability": probabilities.numpy(),
            "predicted_label": (scores > args.threshold).to(torch.int64).numpy(),
        }).to_csv(destination, index=False)
        print(destination)


if __name__ == "__main__":
    main()


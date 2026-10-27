"""Train a new fixed-epoch attention run and export its held-out predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=ROOT / "prepared_samples/SelfAttentionSamples/samples")
    parser.add_argument("--label-dir", type=Path, default=ROOT / "prepared_samples/SelfAttentionSamples/labels")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args(argv)
    if min(args.channels, args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error("channels, epochs, batch size and learning rate must be positive")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if args.output_dir.exists():
        parser.error("Use a new --output-dir; recorded results are never overwritten")

    sys.path.insert(0, str(ROOT / "training_notebooks"))
    from load_data import load_labeled_samples
    try:
        data, columns = load_labeled_samples(args.sample_dir, args.label_dir, args.channels * 40)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset
    sys.path.insert(0, str(ROOT / "TCGA_peaks_prediction"))
    from cross_attention import CrossAttentionModel

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    counts = data["label"].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 2:
        parser.error("At least two peaks of each class are required")
    n = int(counts.min())
    groups = [data[data["label"] == label] for label in [0, 1]]
    balanced = pd.concat([g.sample(n=n, random_state=args.seed) if len(g) > n else g for g in groups])
    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    train_ids, test_ids = train_test_split(
        np.arange(len(balanced)), test_size=args.test_size,
        random_state=args.seed, stratify=balanced["label"],
    )
    x = balanced[columns].to_numpy(dtype=np.float32).reshape(-1, args.channels, 40).transpose(0, 2, 1).copy()
    y = balanced["label"].to_numpy(dtype=np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_ids]), torch.from_numpy(y[train_ids])),
                        batch_size=args.batch_size, shuffle=True)
    model = CrossAttentionModel(input_dim=args.channels, seq_len=40, hidden_dim=64, dropout_rate=0.15)
    model._initialize_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features).reshape(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        history.append({"epoch": epoch + 1, "training_loss": total_loss / len(train_ids)})
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={history[-1]['training_loss']:.6f}")
    # The held-out set is evaluated once, after the fixed final epoch.
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x[test_ids])).reshape(-1)
        probability = torch.sigmoid(logits).numpy()
    predicted = (probability > 0.5).astype(int)
    metrics = {
        "AUROC": float(roc_auc_score(y[test_ids], probability)),
        "AUPRC": float(average_precision_score(y[test_ids], probability)),
        "Accuracy": float(accuracy_score(y[test_ids], predicted)),
        "F1": float(f1_score(y[test_ids], predicted, zero_division=0)),
        "seed": args.seed, "epochs": args.epochs, "channels": args.channels,
        "train_rows": len(train_ids), "test_rows": len(test_ids),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": model.state_dict(), "feature_columns": columns,
                "channels": args.channels, "seed": args.seed}, args.output_dir / "attention_model.pth")
    split = balanced[["source_sample", "sample", "label"]].copy()
    split["split"] = "train"
    split.loc[test_ids, "split"] = "test"
    split.to_csv(args.output_dir / "split.csv", index=False)
    result = split.loc[test_ids].copy()
    result["prediction"] = logits.numpy()
    result["probability"] = probability
    result["predicted_label"] = predicted
    result.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_loss.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

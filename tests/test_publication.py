"""Regression checks for figure source data, sample alignment and input guards."""

import csv
import importlib.util
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_notebooks"))
from load_data import load_labeled_samples


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load_module("plot_fig_s6", ROOT / "figures/S6/plot_fig_s6.py")
predict = load_module("predict", ROOT / "TCGA_peaks_prediction/predict.py")


class FigureSourceTests(unittest.TestCase):
    def test_ml_values_match_saved_evaluations(self):
        comparison = plot.load_comparison().set_index("Model")
        names = {"LightGBM": "LightGBM", "Random Forest": "RandomForest",
                 "SVM": "SVM", "Logistic Regression": "LogisticRegression"}
        for name, directory in names.items():
            text = (ROOT / "saved_models/ML_Model_Output" / directory /
                    "Evaluate_Result_TestSet.txt").read_text()
            expected = float(re.search(r"AUROC=([0-9.eE+-]+)", text).group(1))
            self.assertEqual(float(comparison.loc[name, "AUROC"]), expected)

    def test_ablation_means_and_standard_deviations(self):
        expected = {"C": (0.881987621816922, 0.0253375581546),
                    "D": (0.8550033379030483, 0.020346212917302327),
                    "E": (0.8206825721522317, 0.0165823674),
                    "F": (0.8206797777141748, 0.0165831384)}
        for panel, (mean, sd) in expected.items():
            data = plot.load_ablation(panel)
            self.assertEqual(len(data), 16)
            self.assertAlmostEqual(data.loc[0, "mean_auroc"], mean, places=12)
            self.assertAlmostEqual(data.loc[0, "auroc_sd"], sd, places=7)
            np.testing.assert_allclose(data["auroc_sd"] ** 2, data["auroc_variance"], atol=1e-15)

    def test_error_bars_are_not_clipped(self):
        for panel in "CDEF":
            data = plot.load_ablation(panel)
            fig = plot.plot_ablation(panel)
            low, high = fig.axes[0].get_ylim()
            self.assertGreaterEqual(high, float((data["mean_auroc"] + data["auroc_sd"]).max()))
            self.assertLessEqual(low, float((data["mean_auroc"] - data["auroc_sd"]).min()))
            plot.plt.close(fig)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.samples = Path(self.tmp.name) / "samples"
        self.labels = Path(self.tmp.name) / "labels"
        self.samples.mkdir()
        self.labels.mkdir()

    def write_sample(self, name, values, labels):
        pd.DataFrame({"sample": ["peak_a", "peak_b"], "f0": values}).to_csv(
            self.samples / f"{name}_sample_prepared.csv", index=False)
        pd.DataFrame({"sample": ["peak_b", "peak_a"], "label": labels}).to_csv(
            self.labels / f"{name}_labels.csv", index=False)

    def test_aligns_reordered_labels_within_each_sequencing_sample(self):
        self.write_sample("seq1", [11, 12], [0, 1])
        self.write_sample("seq2", [21, 22], [1, 0])
        data, columns = load_labeled_samples(self.samples, self.labels, expected_features=1)
        self.assertEqual(columns, ["f0"])
        self.assertEqual(len(data), 4)
        actual = data.set_index(["source_sample", "sample"])["label"].to_dict()
        self.assertEqual(actual, {("seq1", "peak_a"): 1, ("seq1", "peak_b"): 0,
                                  ("seq2", "peak_a"): 0, ("seq2", "peak_b"): 1})

    def test_rejects_wrong_feature_count(self):
        self.write_sample("seq1", [1, 2], [0, 1])
        with self.assertRaisesRegex(ValueError, "expected 640 features, found 1"):
            load_labeled_samples(self.samples, self.labels, expected_features=640)

    def test_rejects_unmatched_label_ids(self):
        self.write_sample("seq1", [1, 2], [0, 1])
        pd.DataFrame({"sample": ["peak_a", "missing"], "label": [0, 1]}).to_csv(
            self.labels / "seq1_labels.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            load_labeled_samples(self.samples, self.labels, expected_features=1)


class PredictionInputTests(unittest.TestCase):
    def test_bundled_demo_is_rejected_as_full_feature_input(self):
        path = next((ROOT / "prepared_samples/SelfAttentionSamples/samples").glob("*.csv"))
        with self.assertRaisesRegex(ValueError, "found 520 features"):
            predict.validate_input(path)

    def test_rejects_non_finite_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample", *[f"f{i}" for i in range(40)]])
                writer.writerow(["peak_a", *([0] * 39), "nan"])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                predict.validate_input(path, channels=1)


if __name__ == "__main__":
    unittest.main()

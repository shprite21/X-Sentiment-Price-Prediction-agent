"""Refit four forecasting models offline from a checksummed public bar-level feature snapshot."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from compare_periods import inputs

from xsentiment.config import Config
from xsentiment.model import evaluate, select_model


def verify_models(directory: Path) -> None:
    manifest, frame = inputs(directory)
    cfg = Config(**manifest["config"])
    predictions, folds = evaluate(frame, manifest["price_only_features"], manifest["features"], cfg)
    selected, _ = select_model(predictions)
    keys = ["phase", "fold", "time", "model", "horizon"]
    observed = pd.read_csv(directory / "predictions.csv", parse_dates=["time"])
    left = predictions.set_index(keys).sort_index()
    right = observed.set_index(keys).sort_index()
    pd.testing.assert_index_equal(left.index, right.index)
    np.testing.assert_allclose(
        left[["predicted_return", "actual_return"]],
        right[["predicted_return", "actual_return"]],
        rtol=1e-9,
        atol=1e-12,
    )
    if selected != manifest["selected_model"]:
        raise ValueError("Selected model differs from the recorded validation choice")
    print(f"Verified {len(predictions):,} predictions across {len(folds)} blocks, selected {selected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify_models(parser.parse_args().directory)

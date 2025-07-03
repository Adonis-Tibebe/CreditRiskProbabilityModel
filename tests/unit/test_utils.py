import numpy as np
import pytest
from src.utils.utils import evaluate_model

def test_evaluate_model(monkeypatch):
    # Small valid binary classification example
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.8, 0.2, 0.4])

    # Patch plt.show to avoid opening a window during tests
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    metrics = evaluate_model(y_true, y_pred, y_proba, model_name="TestModel")

    assert isinstance(metrics, dict)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert isinstance(metrics[key], float)
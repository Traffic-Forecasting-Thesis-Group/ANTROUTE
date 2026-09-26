import numpy as np


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(scores))


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean class distance (Light=0, Medium=1, Heavy=2): a Light/Heavy mix-up costs twice a neighbour."""
    return float(np.abs(y_true.astype(int) - y_pred.astype(int)).mean())

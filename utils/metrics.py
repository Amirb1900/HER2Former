import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)


class MetricsCalculator:
    """
    Metrics for HER2Former

    Supports:

        - Accuracy
        - Balanced Accuracy
        - Macro F1
        - Weighted F1
        - Quadratic Weighted Kappa (QWK)

    """

    def __init__(self):
        pass


    def compute(
        self,
        y_true,
        y_pred
    ):

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        results = {

            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    y_true,
                    y_pred
                ),

            "macro_f1":
                f1_score(
                    y_true,
                    y_pred,
                    average="macro"
                ),

            "weighted_f1":
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted"
                ),

            "qwk":
                cohen_kappa_score(
                    y_true,
                    y_pred,
                    weights="quadratic"
                ),

            "confusion_matrix":
                confusion_matrix(
                    y_true,
                    y_pred
                )

        }

        return results

""" Alpine Meadow primitives for predictions. """

import numpy as np
from sklearn.metrics import precision_recall_curve

from .base import BasePrimitive


class ThresholdingPrimitive(BasePrimitive):
    """
    Multiclass thresholding Primitive for F1 scores.
    """

    def __init__(self):
        """
        y_truth: cv true labels
        y_proba_hat: cv probabilities output by the predictor
        y_hat: cv predicted labels by the predictor
        """
        super().__init__()
        self._optimal_thetas = []
        self._train_y_truth = None
        self._train_y_proba_hat = None

    def set_training_data(self, inputs, outputs):  # pylint: disable=arguments-differ
        self._train_y_proba_hat = inputs
        self._train_y_truth = outputs.values
        self._input_columns = list(inputs.columns)
        self._output_columns = list(outputs.columns)

    def _fit(self):
        """Find optimal threshold for F1 score based on the the CV results"""

        # iterate over the columns of probas array to compute optimal class thresholds
        self._optimal_thetas = []
        for class_ in self._train_y_proba_hat.columns:
            class_probas = self._train_y_proba_hat[class_].values
            class_labels = np.array(self._train_y_truth)
            class_labels[self._train_y_truth == class_] = 1
            class_labels[self._train_y_truth != class_] = 0

            precision, recall, thresholds = precision_recall_curve(class_labels, class_probas)

            fscore = (2 * precision * recall) / (precision + recall)
            fscore = np.nan_to_num(fscore, nan=0.0)

            if set(fscore) != {0.0}:
                ix = np.nanargmax(fscore)
                self._optimal_thetas.append(thresholds[ix])
            else:
                # not enough information to compute compute f1 scores
                self._optimal_thetas.append(0.0)

        self._train_y_truth = None
        self._train_y_proba_hat = None

    def fit(self):
        self._fit()

    def produce(self, inputs):  # pylint: disable=arguments-differ
        """ Gets probas as input, outputs classes base on thresholding max_margin_scores."""

        test_y_proba = inputs
        max_margin_scores = (test_y_proba - self._optimal_thetas)
        max_margin_score_classes = max_margin_scores.idxmax(axis=1)
        return max_margin_score_classes

    def produce_proba(self, inputs):  # pylint: disable=arguments-differ
        return inputs

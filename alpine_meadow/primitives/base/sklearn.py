""" Alpine Meadow primitive wrapper for sklearn. """

import numpy as np
import pandas as pd

from .base import BasePrimitive


class SKLearnPrimitive(BasePrimitive):
    """
    Base wrapper class for sklearn primitives.
    """

    def __init__(self, primitive):
        super().__init__()
        self._primitive = primitive
        self._inputs = None
        self._outputs = None
        self._output_columns = None

    def set_training_data(self, inputs, outputs=None):
        self._inputs = inputs
        self._outputs = outputs
        self._input_columns = list(self._inputs.columns)

    def fit(self):
        if self._outputs is None:
            self.primitive.fit(self._inputs)

            from sklearn.impute import SimpleImputer as Imputer

            if isinstance(self.primitive, Imputer):
                all_columns = list(self._inputs.columns)
                non_nan_column_indices = np.argwhere(~np.isnan(self.primitive.statistics_)).flatten()
                non_nan_columns = []
                for non_nan_column_index in non_nan_column_indices:
                    non_nan_columns.append(all_columns[non_nan_column_index])
                self._output_columns = non_nan_columns
            else:
                self._output_columns = list(self._inputs.columns)
        else:
            self.primitive.fit(self._inputs, self._outputs)
            self._output_columns = list(self._outputs.columns)

        self._inputs = None
        self._outputs = None

    def produce(self, inputs):
        if getattr(self.primitive, "predict", None):
            return pd.DataFrame(self.primitive.predict(inputs))
        output = self.primitive.transform(inputs)
        if len(output.shape) == 2:
            if output.shape[1] == len(self._output_columns):
                return pd.DataFrame(output, columns=self._output_columns)
            return pd.DataFrame(output)

        return pd.DataFrame(output)

    def produce_proba(self, inputs):
        if hasattr(self.primitive, "predict_proba"):
            return pd.DataFrame(self.primitive.predict_proba(inputs), columns=self.primitive.classes_)
        raise RuntimeError(f"{self.primitive} cannot predict probabilities!")

    def __repr__(self):
        return self.primitive.__repr__()

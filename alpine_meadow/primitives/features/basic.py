""" Basic feature processing primitives. """

import numpy as np
import pandas as pd

from ..base import BasePrimitive, BaseTransformerPrimitive


class ExtractColumnsByNames(BaseTransformerPrimitive):
    """
    A primitive which extracts the columns from the inputs based on the given names.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._use_columns_names = kwargs.get('names')

    def produce(self, inputs):
        inputs_columns = list(inputs.columns)
        self._input_columns = inputs_columns
        columns = list(filter(lambda column: column in inputs_columns, self._use_columns_names))
        outputs = inputs[columns]

        return outputs


class HorizontalConcat(BaseTransformerPrimitive):
    """
    A primitive which concatenates two DataFrames horizontally.
    """

    def produce(self, left, right):  # pylint: disable=arguments-differ
        self._input_columns = list(left.columns) + list(right.columns)
        self._output_columns = self._input_columns
        outputs = pd.concat([left, right], axis=1)

        return outputs


class TimestampConverter(BasePrimitive):
    """
    Convert timestamp to integers.
    """

    def set_training_data(self, inputs, outputs=None):
        self._inputs = inputs
        self._values = {}
        self._input_columns = list(inputs.columns)
        self._output_columns = self._input_columns

    def fit(self):
        inputs = self._inputs
        for column in inputs.columns:
            self._values[column] = inputs[column].mode()[0]
        self._inputs = None

    def produce(self, inputs):
        outputs = inputs.copy()
        for column in inputs.columns:
            outputs[column] = inputs[column].fillna(self._values[column])
            outputs[column] = outputs[column].astype(np.int)
        return outputs

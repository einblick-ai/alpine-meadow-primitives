# pylint: disable=invalid-name
""" Encoder primitives. """

import pandas as pd
import numpy as np
import scipy.sparse

from alpine_meadow.primitives.base.base import BasePrimitive
from .one_hot_utils import OneHotEncoder as OneHotEncoderImpl


class UnseenLabelEncoder(BasePrimitive):
    """
    Label encoder that can puts any unseen categories into a single category.
    """

    def set_training_data(self, inputs, outputs=None):  # pylint: disable=unused-argument
        self._inputs = inputs
        self._input_columns = list(inputs.columns)
        self._output_columns = self._input_columns

    def fit(self):
        columns_to_use = range(0, len(self._inputs.columns))
        self._labels = {}
        self._inverse_labels = {}
        for column_index in columns_to_use:
            self._fit_column(column_index)

        self._inputs = None

    def _fit_column(self, column_index: int):
        """
        Fit the label encoder on the given column.
        :param column_index:
        :return:
        """

        self._labels[column_index] = {}
        self._inverse_labels[column_index] = {}

        for value in self._inputs.iloc[:, column_index]:
            value = str(value).strip()
            if value not in self._labels[column_index]:
                # We add 1 to reserve 0.
                new_label = len(self._labels[column_index]) + 1
                self._labels[column_index][value] = new_label
                self._inverse_labels[column_index][new_label] = value

    def produce(self, inputs):
        columns_to_use = range(0, len(inputs.columns))
        output_columns = [self._produce_column(inputs, column_index) for column_index in columns_to_use]

        if output_columns:
            outputs = pd.concat(output_columns, axis=1)
        else:
            outputs = pd.DataFrame()

        return outputs

    def _produce_column(self, inputs, column_index):
        column = pd.DataFrame([self._labels[column_index].get(str(value).strip(), 0)
                               for value in inputs.iloc[:, column_index]],
                              columns=[inputs.columns[column_index]])

        return column


class OneHotEncoder(BasePrimitive):
    """
    One hot encoder with minimum fraction.
    Adopted from https://github.com/automl/auto-sklearn/blob/master/autosklearn/
    pipeline/components/data_preprocessing/one_hot_encoding/one_hot_encoding.py
    """

    def __init__(self, use_minimum_fraction=True, minimum_fraction=0.01):
        super().__init__()
        self.use_minimum_fraction = use_minimum_fraction
        self.minimum_fraction = minimum_fraction

    def set_training_data(self, inputs, outputs=None):  # pylint: disable=unused-argument
        self._inputs = inputs
        self._input_columns = list(inputs.columns)
        self._output_columns = self._input_columns

    def _fit(self, X):
        """
        Fit the one hot encoder.
        """

        if self.use_minimum_fraction is False:
            self.minimum_fraction = None
        else:
            self.minimum_fraction = float(self.minimum_fraction)

        self.preprocessor = OneHotEncoderImpl(minimum_fraction=self.minimum_fraction,
                                              sparse=False)
        return self.preprocessor.fit_transform(X)

    def fit(self):
        self._fit(self._inputs)
        self._inputs = None

    def produce(self, inputs):  # pylint: disable=missing-function-docstring
        X = inputs
        is_sparse = scipy.sparse.issparse(X)
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X)
        if is_sparse:
            return X
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        return pd.DataFrame(X.toarray())

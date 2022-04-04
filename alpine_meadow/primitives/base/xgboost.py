""" Alpine Meadow primitive wrapper for xgboost. """

import pandas as pd

from .sklearn import SKLearnPrimitive


class XGBoostPrimitive(SKLearnPrimitive):
    """
    Base wrapper class for xgboost primitives.
    """

    def set_training_data(self, inputs, outputs):  # pylint: disable=signature-differs
        if isinstance(inputs, pd.DataFrame):
            self._input_columns = list(inputs.columns)
            inputs = inputs.values

        self._inputs = inputs
        self._outputs = outputs
        self._output_columns = list(self._outputs.columns)

    def produce(self, inputs):
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.values

        return pd.DataFrame(self.primitive.predict(inputs), columns=self._output_columns)

    def produce_proba(self, inputs):
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.values

        return pd.DataFrame(self.primitive.predict_proba(inputs), columns=self.primitive.classes_)

    def __repr__(self):
        return self.primitive.__repr__()

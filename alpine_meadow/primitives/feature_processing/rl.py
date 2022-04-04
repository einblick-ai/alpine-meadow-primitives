""" Reinforcement learning-based feature engineering primitive. """

from ..base import BaseTransformerPrimitive


class FeatureEngineering(BaseTransformerPrimitive):
    """
    A primitive which adds features to a DataFrame given appropriate parameter inputs.
    """

    def __init__(self, features):
        super().__init__()
        self._features = features

    def produce(self, inputs):
        from rl_feature_eng import Primitive as fe_primitive

        self._input_columns = list(inputs.columns)
        return fe_primitive.transform(inputs, names=self._features)

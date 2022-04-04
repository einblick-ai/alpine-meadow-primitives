# pylint: disable=invalid-name
"""Alpine Meadow primitive base class."""

from abc import ABC, abstractmethod


class BasePrimitive(ABC):
    """
    The base class for utils, here we adopt the design of D3M utils
    """

    def __init__(self):
        self._primitive = self
        self._input_columns = None
        self._output_columns = None

    @property
    def primitive(self):
        return self._primitive

    @property
    def input_columns(self):
        return self._input_columns

    @property
    def output_columns(self):
        return self._output_columns

    @abstractmethod
    def set_training_data(self, inputs, outputs):
        pass

    @abstractmethod
    def fit(self):
        """ Fit the primitive if applicable. """

    @abstractmethod
    def produce(self, inputs):
        """ Produce outputs based on input data. """


class BaseTransformerPrimitive(BasePrimitive):  # pylint: disable=abstract-method
    """
    The base class for transformer utils, which doesn't need to be trained
    """

    def set_training_data(self, *args, **kwargs):  # pylint: disable=arguments-differ
        pass

    def fit(self):
        pass

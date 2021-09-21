from abc import ABC
from sklearn import metrics

from utils.exceptions import ImproperlyConfigured
from utils.models import Model


class Metric(ABC):
    _metric = None

    def __init__(self, model: Model, validation_y, test_y):
        if self._metric is None:
            raise ImproperlyConfigured('A metric needs to be specified.')

        self._model = model
        self._validation_y = validation_y
        self._test_y = test_y

    def __str__(self):
        return f'{self._model.__class__.__name__}:\n' \
               f'{self.__class__.__name__} validation: {round(getattr(metrics, self._metric)(self._validation_y, self._model.validation_predictions), 2)}\n' \
               f'{self.__class__.__name__} test: {round(getattr(metrics, self._metric)(self._test_y, self._model.test_predictions), 2)}'

    @property
    def model(self):
        return self._model

    @property
    def validation_y(self):
        return self._validation_y

    @property
    def test_y(self):
        return self._test_y

    @property
    def validation_score(self):
        return self._metric(self._validation_y, self._model.validation_predictions)

    @property
    def test_score(self):
        return self._metric(self._test_y, self._model.test_predictions)


class Accuracy(Metric):
    _metric = 'accuracy_score'


class Recall(Metric):
    _metric = 'recall_score'


class Precision(Metric):
    _metric = 'precision_score'


class F1Score(Metric):
    _metric = 'f1_score'

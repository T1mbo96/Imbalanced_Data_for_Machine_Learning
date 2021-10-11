import pandas as pd

from utils.models import XGBoostClassification, RandomForestClassification, LogisticRegressionClassification
from utils.metrics import Accuracy, BalancedAccuracy, Recall, Precision, F1Score, \
    AreaUnderTheReceiverOperatingCharacteristicCurve, GeometricMean


class TrainEvaluation:
    _allowed_models = {
        'xgboost': XGBoostClassification,
        'random_forest': RandomForestClassification,
        'logistic_regression': LogisticRegressionClassification
    }

    _allowed_metrics = {
        'accuracy': Accuracy,
        'balanced_accuracy': BalancedAccuracy,
        'recall': Recall,
        'precision': Precision,
        'f1_score': F1Score,
        'roc_auc_score': AreaUnderTheReceiverOperatingCharacteristicCurve,
        'geometric_mean_score': GeometricMean
    }

    def __init__(self, train_x: pd.DataFrame, train_y: pd.DataFrame, validation_x: pd.DataFrame,
                 validation_y: pd.DataFrame, test_x: pd.DataFrame, test_y: pd.DataFrame, models, metrics):
        if not isinstance(models, (list, tuple, set)):
            raise ValueError('Models needs to be a list, tuple or set.')

        if not isinstance(metrics, (list, tuple, set)):
            raise ValueError('Metrics needs to be a list, tuple or set.')

        if any(model not in self._allowed_models.keys() for model in models):
            raise ValueError(f'Allowed models: {self._allowed_models}.')

        if any(metric not in self._allowed_metrics.keys() for metric in metrics):
            raise ValueError(f'Allowed metrics: {self._allowed_metrics}.')

        self._train_x = train_x
        self._train_y = train_y
        self._validation_x = validation_x
        self._validation_y = validation_y
        self._test_x = test_x
        self._test_y = test_y

        self._models = self._generate_models(models=set((model.lower() for model in models)))
        self._metrics = self._generate_metrics(metrics=set((metric.lower() for metric in metrics)))

    def __str__(self):
        representation: str = ''

        for _result in self._metrics:
            representation += str(_result) + '\n'

        return representation

    def __repr__(self):
        return self.__str__()

    def _generate_models(self, models):
        return [
            self._allowed_models[_model](train_x=self._train_x, train_y=self._train_y, validation_x=self._validation_x,
                                         test_x=self._test_x) for _model in models]

    def _generate_metrics(self, metrics):
        _results = []

        for _model in self._models:
            for _metric in metrics:
                _results.append(
                    self._allowed_metrics[_metric](model=_model, validation_y=self._validation_y, test_y=self._test_y))

        return _results

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_y(self):
        return self._train_y

    @property
    def validation_x(self):
        return self._validation_x

    @property
    def validation_y(self):
        return self._validation_y

    @property
    def test_x(self):
        return self._test_x

    @property
    def test_y(self):
        return self._test_y

    @property
    def models(self):
        return self._models

    @property
    def metrics(self):
        return self._metrics

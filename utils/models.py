import pandas as pd
import xgboost as xgb

from abc import ABC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils.exceptions import ImproperlyConfigured


class Model(ABC):
    _model = None

    def __init__(self, train_x: pd.DataFrame, train_y: pd.DataFrame, validation_x: pd.DataFrame, test_x: pd.DataFrame):
        if self._model is None:
            raise ImproperlyConfigured('A model needs to be specified.')

        self._train_x = train_x
        self._train_y = train_y
        self._validation_x = validation_x
        self._test_x = test_x

        self._train()

    def _train(self):
        self._model.fit(self._train_x, self._train_y)

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
    def test_x(self):
        return self._test_x

    @property
    def validation_predictions(self):
        return self._model.predict(self._validation_x)

    @property
    def test_predictions(self):
        return self._model.predict(self._test_x)


class XGBoostClassification(Model):
    _model = xgb.XGBClassifier(eval_metric='error', use_label_encoder=False)

    def __str__(self):
        return '<XGBoost Classifier>'


class RandomForestClassification(Model):
    _model = RandomForestClassifier()

    def __str__(self):
        return '<Random Forest Classifier>'

    def _train(self):
        self._model.fit(self._train_x, self._train_y.to_numpy().flatten())


class LogisticRegressionClassification(Model):
    _model = LogisticRegression(max_iter=10000)

    def __str__(self):
        return '<Logistic Regression Classifier>'

    def _train(self):
        self._model.fit(self._train_x, self._train_y.to_numpy().flatten())

import pandas as pd
import xgboost as xgb
import numpy as np

from abc import ABC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV

from utils.exceptions import ImproperlyConfigured


class Model(ABC):
    _model = None
    _parameter_distributions = None

    def __init__(self, train_x: pd.DataFrame, train_y: pd.DataFrame, validation_x: pd.DataFrame, test_x: pd.DataFrame):
        if self._model is None:
            raise ImproperlyConfigured('A model needs to be specified.')

        self._train_x = train_x
        self._train_y = train_y
        self._validation_x = validation_x
        self._test_x = test_x

        self._train()

    def _best_params(self):
        rscv = RandomizedSearchCV(self._model, self._parameter_distributions, verbose=3, n_jobs=-1)
        rscv.fit(self._train_x, self._train_y.to_numpy().flatten())

        return rscv.best_params_

    def _train(self):
        if self._parameter_distributions is not None:
            self._model.set_params(**self._best_params())

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

    @property
    def test_predictions_probability(self):
        return self._model.predict_proba(self._test_x)[:, 1]


class XGBoostClassification(Model):
    _model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='error', use_label_encoder=False)
    _parameter_distributions = {
        'max_depth': [max_depth for max_depth in range(10, 15, 1)],
        'min_child_weight': [min_child_weight for min_child_weight in range(1, 9, 1)],
        'subsample': [subsample for subsample in np.linspace(0.5, 1, 6)],
        'colsample_bytree': [colsample_bytree for colsample_bytree in np.linspace(0.5, 1, 6)],
        'lambda': [lambda_xg for lambda_xg in np.linspace(0.5, 1, 6)],
        'alpha': [alpha for alpha in np.linspace(0, 1, 2)],
        'eta': [round(eta, 2) for eta in np.linspace(0.05, 0.3, 6)],
        'n_estimators': [n_estimators for n_estimators in range(100, 600, 100)]
    }

    def __str__(self):
        return '<XGBoost Classifier>'


class RandomForestClassification(Model):
    _model = RandomForestClassifier()
    _parameter_distributions = {
        'max_depth': [max_depth for max_depth in range(10, 15, 1)],
        'min_samples_leaf': [min_samples_leaf for min_samples_leaf in range(1, 9, 1)],
        'min_samples_split': [min_samples_split for min_samples_split in range(1, 9, 1)],
        'n_estimators': [n_estimators for n_estimators in range(100, 600, 100)]
    }

    def __str__(self):
        return '<Random Forest Classifier>'

    def _train(self):
        self._model.fit(self._train_x, self._train_y.to_numpy().flatten())


class LogisticRegressionClassification(Model):
    _model = LogisticRegressionCV(max_iter=10000)

    def __str__(self):
        return '<Logistic Regression Classifier>'

    def _train(self):
        self._model.fit(self._train_x, self._train_y.to_numpy().flatten())

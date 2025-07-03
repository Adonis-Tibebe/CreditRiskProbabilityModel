import numpy as np
import pandas as pd
import pytest
from src.models.model_trainer import ModelTrainer

def make_data(n=8):
    X = pd.DataFrame({
        "a": np.random.randn(n),
        "b": np.random.randn(n)
    })
    y = np.random.randint(0, 2, size=n)
    return X, y

def test_logistic_regression_train_predict():
    X, y = make_data()
    trainer = ModelTrainer("logistic_regression")
    model = trainer.train(X, y)
    preds = trainer.predict(X)
    assert len(preds) == len(y)
    probas = trainer.predict_proba(X)
    assert np.all((probas >= 0) & (probas <= 1))

def test_random_forest_train_predict():
    X, y = make_data()
    trainer = ModelTrainer("random_forest")
    model = trainer.train(X, y)
    preds = trainer.predict(X)
    assert len(preds) == len(y)

def test_gbm_train_predict():
    X, y = make_data()
    trainer = ModelTrainer("gbm")
    model = trainer.train(X, y)
    preds = trainer.predict(X)
    assert len(preds) == len(y)

def test_grid_search_resource_light():
    X, y = make_data()
    trainer = ModelTrainer("random_forest")
    param_grid = {"n_estimators": [5]}  # Only one value
    model = trainer.train(X, y, param_grid=param_grid, search_type="grid")
    preds = trainer.predict(X)
    assert len(preds) == len(y)

def test_random_search_resource_light():
    X, y = make_data()
    trainer = ModelTrainer("gbm")
    param_grid = {"n_estimators": [5, 10]}  # Only two values
    model = trainer.train(X, y, param_grid=param_grid, search_type="random")
    preds = trainer.predict(X)
    assert len(preds) == len(y)

def test_invalid_model_name():
    with pytest.raises(ValueError):
        ModelTrainer("not_a_model")
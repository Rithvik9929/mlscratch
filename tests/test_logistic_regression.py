import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import train_test_split

from mlscratch.logistic_regression import LogisticRegression


@pytest.fixture
def data():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)

def test_output_shapes(data):
    X_train, X_test, y_train, _ = data
    model = LogisticRegression().fit(X_train, y_train)
    assert model.predict_probabilities(X_test).shape == (len(X_test),)
    assert model.predict(X_test).shape == (len(X_test),)

def test_probabilities_valid(data):
    X_train, X_test, y_train, _ = data
    proba = LogisticRegression().fit(X_train, y_train).predict_probabilities(X_test)
    assert np.all((proba >= 0) & (proba <= 1))

def test_accuracy_close_to_sklearn(data):
    X_train, X_test, y_train, y_test = data
    implemented_model = LogisticRegression(lr=0.1, n_iters=5000).fit(X_train, y_train)
    sk_model = SkLogisticRegression(penalty=None).fit(X_train, y_train)
    assert abs(implemented_model.score(X_test, y_test) - sk_model.score(X_test, y_test)) <= 0.02    

def test_probabilities_close_to_sklearn(data):
    X_train, X_test, y_train, _ = data
    implemented_model  = LogisticRegression(lr=0.1, n_iters=20000).fit(X_train, y_train)
    sk_model = SkLogisticRegression(penalty=None).fit(X_train, y_train)
    my_predictions = implemented_model.predict_probabilities(X_test)
    sklearn_predictions = sk_model.predict_proba(X_test)[:, 1]
    np.testing.assert_allclose(my_predictions,sklearn_predictions,atol=0.05)


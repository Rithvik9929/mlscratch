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

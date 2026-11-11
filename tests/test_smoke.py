import numpy as np
import sklearn

import mlscratch


def test_environment():
    assert mlscratch is not None
    assert np.__version__
    assert sklearn.__version__
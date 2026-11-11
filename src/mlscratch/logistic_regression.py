import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1+ np.exp(-z))


class LogisticRegression:
    def __init__(self, lr: float = 0.1, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        n_samples = X.shape[0]

        for _ in range(self.n_iters):
            h = (X @ self.weights) + self.bias 
            results = _sigmoid(h)

            gradient_weights = (-1/n_samples) * (X.T @ (Y-results))
            gradient_bias = (-1/n_samples) * np.sum(Y-results)

            self.weights = self.weights - (self.lr * gradient_weights)
            self.bias = self.bias - (self.lr * gradient_bias)

        return self
        
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self.predict_probabilities(X)
        return np.where(predictions > 0.5, 1, 0)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        predictions = _sigmoid((X @ self.weights) + self.bias)
        return predictions

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        predictions = self.predict(X)
        accuracy = 0
        n = len(Y)
        for i in range(len(Y)):
            if(predictions[i] == Y[i]):
             accuracy += 1/n   
        return accuracy



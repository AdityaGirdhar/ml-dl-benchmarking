import numpy as np
class LinearRegression:
    # OLS Method (Ordinary Least Squares) training
    def OLS_fit(self, X, y):
        # Add bias
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        # Calculate weights
        weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return weights
    def OLS_predict(self, X, weights):
        # Add bias
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        # Calculate predictions
        predictions = X.dot(weights)
        return predictions
    def score(self,y, predictions):
        mse = np.mean((y - predictions)**2)
        accuracy = 1 - mse / np.var(y)
        return mse, accuracy
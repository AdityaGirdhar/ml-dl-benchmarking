import numpy as np
from mxnet import nd
class LinearRegression:
    # OLS Method (Ordinary Least Squares) training
    def OLS_fit(self, X, y):
        # adding a column of ones to X
        X = nd.concat(X, nd.ones((X.shape[0], 1)), dim=1)
        # calculating the weights
        product = nd.linalg_gemm2(X, X, transpose_a=True)
        inverse = nd.linalg_inverse(product)
        product2 = nd.linalg_gemm2(X, y, transpose_a=True)
        w = nd.linalg_gemm2(inverse, product2)
        return w
    def OLS_predict(self, X, w):
        # adding a column of ones to X
        X = nd.concat(X, nd.ones((X.shape[0], 1)), dim=1)
        # calculating the predictions
        y_pred = nd.linalg_gemm2(X, w)
        y_pred = y_pred.reshape((y_pred.shape[0], 1))
        return y_pred
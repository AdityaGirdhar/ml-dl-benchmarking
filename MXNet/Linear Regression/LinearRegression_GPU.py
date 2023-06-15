from mxnet.gluon import nn
from mxnet import autograd
from mxnet import gluon
import mxnet as mx
from mxnet import nd
data_ctx = mx.gpu()
model_ctx = mx.gpu()
class LinearRegression:
    def OLS_fit(self, X, y):
        # adding a column of ones to X
        X = nd.concat(X, nd.ones((X.shape[0], 1), ctx=data_ctx), dim=1)
        product = nd.linalg_gemm2(X, X, transpose_a=True)
        inverse = nd.linalg_inverse(product)
        product2 = nd.linalg_gemm2(X, y, transpose_a=True)
        w = nd.linalg_gemm2(inverse, product2)
        return w
    def OLS_predict(self, X, w):
        # adding a column of ones to X
        X = nd.concat(X, nd.ones((X.shape[0], 1), ctx=data_ctx), dim=1)
        y_pred = nd.linalg_gemm2(X, w)
        y_pred = y_pred.reshape((y_pred.shape[0], 1))
        return y_pred
    
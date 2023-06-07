from mxnet import nd
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import gluon
from mxnet import np, npx
npx.set_np()
class PrincipalComponentAnalysis:
    def __init__(self, n_components, X):
        # initialize the number of principal components and the features
        self.n_components = n_components
        self.X = X
    def standardize(self):
        # standardize the data
        mean = np.mean(self.X , axis = 0, keepdims = True)
        std = np.std(self.X, axis = 0, keepdims = True)
        self.X = (self.X - mean) / std
    def covariance_matrix(self):
        # calculate the covariance matrix
        self.cov = np.cov(self.X, rowvar = False, bias = False)
    def eigen_decomposition(self):
        # perform eigen decomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov)
    def fit(self):
        self.standardize()
        self.covariance_matrix()
        self.eigen_decomposition()
    def transform(self):
        # sort eigenvalues and eigenvectors in descending order
        self.eigenvalues = self.eigenvalues[::-1]
        self.eigenvectors = self.eigenvectors[:, ::-1]
        # select the first n_components eigenvectors
        self.eigenvectors = self.eigenvectors[:, :self.n_components]
        # transform the data
        self.X_transformed = np.dot(self.X, self.eigenvectors)
        # return the transformed data
        return self.X_transformed
    def fit_transform(self):
        # fit and transform the data
        self.fit()
        return self.transform()

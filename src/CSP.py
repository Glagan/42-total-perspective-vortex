import numpy as np
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator


class CSP(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=4):
        if not isinstance(n_components, int):
            raise ValueError("n_components must be an integer.")
        self.n_components = n_components

    def _compute_covariance_matrices(self, X, y):
        _, n_channels, _ = X.shape

        # Compute the covariance matrix on each channels
        covs = []
        for cur_class in self._classes:
            x_class = X[y == cur_class]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(n_channels, -1)
            cov_mat = np.cov(x_class)
            covs.append(cov_mat)

        return np.stack(covs)

    def fit(self, X, y):
        self._classes = np.unique(y)

        covs = self._compute_covariance_matrices(X, y)
        eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))

        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        eigen_vectors = eigen_vectors[:, ix]

        # Create a filter on the number of output dimensions
        self.filters_ = eigen_vectors.T
        pick_filters = self.filters_[: self.n_components]

        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        X = (X**2).mean(axis=2)

        return self

    def transform(self, X):
        # "Crop" to the amount of dimensions from the fitted filter
        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
        X = (X**2).mean(axis=2)  # Variance
        X -= X.mean()
        X /= X.std()
        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

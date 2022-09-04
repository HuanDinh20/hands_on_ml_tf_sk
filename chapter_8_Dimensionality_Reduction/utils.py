from sklearn.decomposition import PCA
import numpy as np


def choose_pca_dimension(x, threshold, pca):
    """
    computes PCA without reducing dimensionality,
    preserve 95% of the training set’s variance
    """
    pca.fit(x)
    cum_sum = np.cumsum(pca.explained_variance_ratio)
    n_components = np.argmax(cum_sum >= threshold) + 1
    return n_components






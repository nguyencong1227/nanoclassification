from sklearn.decomposition import PCA
from utils import *
import numpy as np
from sklearn.svm import SVC

def U_for_pca(X, X_mean, n_components=1):
    X_hat = X - X_mean
    pca = PCA(n_components=n_components)
    pca.fit(X_hat)
    U = pca.components_.T
    return U

def pca(X, X_mean, U):
    X_hat = X - X_mean
    X_PCA = np.dot(U.T, X_hat.T).T
    return X_PCA

def extract_PCA(X, X_mean, n_components=3):
    pca = PCA(n_components=n_components)
    X_PCA = pca.fit_transform(X - X_mean)
    return X_PCA


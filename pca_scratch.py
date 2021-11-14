import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        #Covariance (needs samples as columns)
        cov = np.cov(X.T)

        #Eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        #Transpose for easier calculations
        eigenvectors = eigenvectors.T
        
        #Sort Eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        #Store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        #Project data
        X = X - self.mean 
        return np.dot(X, self.components.T)

#Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
x_projected = pca.transform(X)
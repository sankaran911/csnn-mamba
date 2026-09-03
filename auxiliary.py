from sklearn.decomposition import PCA
import numpy as np

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))

    pca = PCA(
        n_components=numComponents,
        whiten=True
    )

    newX = pca.fit_transform(newX)

    newX = np.reshape(
        newX,
        (X.shape[0], X.shape[1], numComponents)
    )

    return newX
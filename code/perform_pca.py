from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as scaler


def perform_pca(data,components):
    # centering. each  mean= 0 and each sd =1
    scaled_data = scaler().fit_transform(data)
    # transpose (T). scale expects samples to be in rows, not col.

    pca = PCA(n_components=components)  # creat PCA-object that can be applied for each dataset.
    principal_components = pca.fit_transform(scaled_data)

    return principal_components

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly
from lib.utils.clustering import KNN
import os
import numpy as np
import torch

def get_pca(X, y, n=2, show_image=False, filename='pca'):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    if n == 2:
        fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=y)
        fig.update_layout(
            title="PCA visualization of Hip Dataset's ViTMAE image features",
            xaxis_title="First Principal Component",
            yaxis_title="Second Principal Component",
        )
    elif n == 3:
        fig = px.scatter_3d(x=X_pca[:,0], y=X_pca[:,1], z=X_pca[:,2], color=y)
        fig.update_layout(title="PCA visualization of Hip Dataset's ViTMAE image features")

    if show_image:
        fig.show()
    else:
        plotly.offline.plot(fig, filename=f'{filename}')
    

def get_tsne(X, y, n=2, show_image=False, filename='tsne'):
    tsne = TSNE(n_components = n, perplexity=100, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(tsne.kl_divergence_)

    if n == 2:
        fig = px.scatter(x=X_tsne[:,0], y=X_tsne[:,1], color=y)
        fig.update_layout(title="t-SNE visualization of Hip Dataset's ViTMAE image features",
                        xaxis_title="First t-SNE",
                        yaxis_title="Second t-SNE",
        )
    elif n == 3:
        fig = px.scatter_3d(x=X_tsne[:,0], y=X_tsne[:,1], z=X_tsne[:,2], color=y)
        fig.update_layout(title="t-SNE visualization of Hip Dataset's ViTMAE image features")
    
    if show_image: fig.show()
    else: plotly.offline.plot(fig, filename=f'{filename}')

def get_umap(X, y, n=2, n_neighbors=15, min_dist=0.1, seed=0, show_image=False, model='ViTMAE', filename='umap'):
    # Ensure data is a CPU numpy array (UMAP/sklearn expects numpy, not torch tensors)
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    else:
        X = np.asarray(X)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    reducer.fit(X)
    embedding = reducer.transform(X)

    # Convert numeric labels to categorical class names for a discrete legend
    label_map = {0: 'dysplastic', 1: 'normal'}
    try:
        y_categorical = [label_map.get(int(lbl), str(lbl)) for lbl in list(y)]
    except Exception:
        # Fallback: if y isn't iterable or conversion fails, just use y directly
        y_categorical = y

    # Use a discrete color legend instead of a continuous colorscale
    fig = px.scatter(
        x=embedding[:,0],
        y=embedding[:,1],
        color=y_categorical,
        color_discrete_map={'dysplastic': 'blue', 'normal': 'red'},
        labels={'color': 'class'}
    )
    fig.update_layout(title=f"UMAP visualization of {model} image features",
                      xaxis_title="First UMAP",
                      yaxis_title="Second UMAP",
    )
    dir = '/'.join(filename.split('/')[:-1])
    if not os.path.exists(dir): os.makedirs(dir)
    if show_image: fig.show()
    else: plotly.offline.plot(fig, filename=f'{filename}')
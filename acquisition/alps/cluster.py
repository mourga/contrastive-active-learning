"""
Code from https://github.com/forest-snow/alps
"""
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans

def closest_center_dist(X, centers):
    # return distance to closest center
    dist = torch.cdist(X, X[centers])
    cd = dist.min(axis=1).values
    return cd


def kmeans_pp(X, k, centers, **kwargs):
    # kmeans++ algorithm
    if len(centers) == 0:
        # randomly choose first center
        c1 = np.random.choice(X.size(0))
        centers.append(c1)
        k -= 1
    # greedily choose centers
    for i in tqdm(range(k)):
        dist = closest_center_dist(X, centers) ** 2
        prob = (dist / dist.sum()).cpu().detach().numpy()
        ci = np.random.choice(X.size(0), p=prob)
        centers.append(ci)
    return centers


def kmeans(X, k, tol=1e-4, **kwargs):
    # kmeans algorithm
    print("Running Kmeans")
    kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(X)
    centers = kmeans.cluster_centers_
    # find closest point to centers
    centroids = cdist(centers, X).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(X)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis = None)
    return centroids

def stopping(lamda, centers):
    k = len(centers)
    return 16 * lamda * k * (np.log2(k)+ 2)

def kcenter(X, k, centers, **kwargs):
    if len(centers) == 0:
        # randomly choose first center
        c1 = np.random.choice(X.size(0))
        centers.append(c1)
        k -= 1
    # greedily choose other centers
    for i in tqdm(range(k)):
        dist = closest_center_dist(X, centers)
        ci = dist.argmax().item()
        centers.append(ci)
    return centers


def badge(grads, k, **kwargs):
    # BADGE algorithm (Ash et al. 2020)
    centers = kmeans_pp(grads, k, [])
    return centers

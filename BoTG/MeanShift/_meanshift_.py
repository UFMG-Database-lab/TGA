
from scipy.sparse.linalg import norm

from tqdm import tqdm

import numpy as np

from sklearn.utils import gen_batches
from sklearn.neighbors import NearestNeighbors

#from joblib import Parallel, delayed
from sklearn.externals.joblib import Parallel, delayed

def __mean_shift_single_seed__(idx_seed, X, nbrs, max_iter, verbose=False):
    atual_point = X[idx_seed]
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    completed_iterations = 0
    while True:
        points_within = nbrs.radius_neighbors([atual_point], return_distance=False)[0]
        if len(points_within) == 0:
            idx_news = -1
            points_within = []
            break
        old_point = atual_point
        atual_point = X[points_within].mean(axis=0)
        thresh = np.linalg.norm(old_point - atual_point)
        if (thresh < stop_thresh or completed_iterations == max_iter):
            idx_news      = nbrs.kneighbors([atual_point], n_neighbors=1, return_distance=False)[0][0]
            points_within = nbrs.radius_neighbors([X[idx_news]], return_distance=False)[0]
            break
        completed_iterations += 1
    return idx_news, set(points_within)

def build_clusters(X, n_jobs=8, max_iter=100, metric='cosine', quantile=0.001, verbose=False):
    bandwidth = estimate_bandwidth(X, n_jobs, metric=metric, quantile=quantile, verbose=verbose)
    
    if bandwidth <= 0:
        bandwidth = 1.

    seeds = estimate_seed(X)

    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1, metric=metric).fit(X)

    all_res = Parallel(n_jobs=n_jobs)(delayed(__mean_shift_single_seed__) (idx_seed, X, nbrs, max_iter) for idx_seed in tqdm(seeds, position=1, desc="Estimating clusters", disable=not verbose))
    
    center_intensity_dict = [ (i, res) for i, res in enumerate(all_res) if res[0] >= 0 ]
    
    if not center_intensity_dict:
        raise ValueError("No point was within bandwidth=%f of any seed. Try a different seeding strategy or increase the bandwidth." % bandwidth)
    
    clusters = dict([ cluster for id_subgraph, cluster in center_intensity_dict ])
    mapper_subgraph_cluster = dict([ (id_subgraph, id_cluster) for id_subgraph, (id_cluster,_) in center_intensity_dict ])

    clusters, mapper_subgraph_cluster = __dedup_clusters__(X, nbrs, clusters, mapper_subgraph_cluster, bandwidth, max_iter, metric)

    result = { 'bandwidth': bandwidth, 'clusters':clusters, 'mapper_cluster': mapper_subgraph_cluster }

    return result
def __dedup_clusters__(X, nbrs, clusters, mapper_subgraph_cluster, bandwidth, max_iter=100, metric='cosine'):
    updated = True
    completed_iterations = 0
    while updated and completed_iterations < max_iter:
        updated = False
        sorted_clusters = sorted( clusters.items(), key=lambda tup: (len(tup[1]), tup[0]), reverse=True )
        unique = np.ones(len(sorted_clusters), dtype=np.bool)
        centers = X[np.array([cluster_id for cluster_id,_ in sorted_clusters])]
        nbrs_dedup = NearestNeighbors(radius=bandwidth, n_jobs=1, metric=metric).fit(centers)
        graph_of_dist = nbrs_dedup.radius_neighbors_graph(centers)
        available = [ (i, points) for (i, points) in list(enumerate(graph_of_dist)) if np.sum(points.A) > 1 ]
        for (i, points) in available:
            if unique[i]:
                updated = True
                cluster_id, points_within = sorted_clusters[i]
                idxs_points_to_dedup = [ point_id for point_id in np.where(points.A)[1] if point_id != i ]
                for idx_point_to_dedup in idxs_points_to_dedup:
                    idx_cluster_to_dedup = sorted_clusters[idx_point_to_dedup][0]
                    unique[idx_point_to_dedup] = 0
                    if idx_cluster_to_dedup in clusters:
                        points_within = points_within.union( clusters[idx_cluster_to_dedup] )
                        del clusters[idx_cluster_to_dedup]
                    
                mean_point = X[np.array(list(points_within))].mean(axis=0)
                idx_news = nbrs.kneighbors([mean_point], n_neighbors=1, return_distance=False)[0][0]
                unique[i] = idx_news == cluster_id
                if not unique[i]:
                    del clusters[cluster_id]
                    clusters[idx_news] = points_within
                for subgraph_id in points_within:
                    mapper_subgraph_cluster[subgraph_id] = idx_news
        completed_iterations += 1
    return clusters, mapper_subgraph_cluster
def estimate_seed(X):
    return range(X.shape[0])
def estimate_bandwidth(X, n_jobs, metric='cosine', quantile=0.001, verbose=False):
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:  # cannot fit NearestNeighbors with n_neighbors = 0
        n_neighbors = 2
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs, metric=metric).fit(X)
    #proportion = 10
    proportion = min(X.shape[0], 10)
    bandwidth = 0.
    for batch in tqdm(gen_batches(X.shape[0], max(1, X.shape[0]//proportion)), total=proportion, position=1, desc="Estimating bandwidth", disable=not verbose):
        xx = X[batch]
        if not xx.shape[0]:
            break
        d, _ = nbrs.kneighbors(xx, return_distance=True)
        bandwidth += np.max(d, axis=1).sum()
    bandwidth /= X.shape[0]
    if bandwidth <= 0:
        print(X)
        return 1.
    return bandwidth

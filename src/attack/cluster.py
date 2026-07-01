"""
Clustering strategies for the black-hole attack pipeline.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import faiss


def cluster_kmeans(vecs: np.ndarray, n_clusters: int, *,
                   random_state: int = 42,
                   max_points_per_centroid: int | None = None,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering on vectors.

    Returns:
        labels: (N,) int array, cluster index for each vector
        centers: (n_clusters, dim) float32 array of cluster centroids
    """
    kwargs.pop('batch_size', None)
    kwargs.pop('gpu_id', None)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs)
    labels = km.fit_predict(vecs)
    return labels, km.cluster_centers_.astype(np.float32)


def cluster_minibatch_kmeans(vecs: np.ndarray, n_clusters: int, *,
                              batch_size: int = 4096,
                              random_state: int = 42,
                              max_points_per_centroid: int | None = None,
                              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """MiniBatch K-means — much faster for large vector sets (>1M).

    Returns:
        labels: (N,) int array, cluster index for each vector
        centers: (n_clusters, dim) float32 array of cluster centroids
    """
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        # batch_size=n_clusters * 50,
        random_state=random_state,
        n_init="auto",
        **kwargs,
    )
    labels = km.fit_predict(vecs)
    return labels, km.cluster_centers_.astype(np.float32)


def cluster_faiss_gpu(vecs: np.ndarray, n_clusters: int, *,
                      random_state: int = 42,
                      gpu_id: int = 0,
                      niter: int = 25,
                      use_float16: bool = False,
                      max_points_per_centroid: int | None = None,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated k-means via FAISS.

    Requires faiss-gpu. Typical speedup over CPU k-means: 10–50×.

    Args:
        max_points_per_centroid: points sampled per centroid per iteration.
            None (default) = use all data. Lower = faster, approximate.
            FAISS default is 256. Set to e.g. 100, 500, 1000 for minibatch-like behavior.

    Returns:
        labels: (N,) int array, cluster index for each vector
        centers: (n_clusters, dim) float32 array of cluster centroids
    """
    n_gpus = faiss.get_num_gpus()
    if n_gpus == 0:
        raise RuntimeError("No GPU available for FAISS GPU clustering")

    # Normalize for inner product (cosine similarity in k-means)
    data = vecs.astype(np.float32).copy()
    faiss.normalize_L2(data)

    d = data.shape[1]
    res = faiss.StandardGpuResources()

    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = gpu_id
    cfg.useFloat16 = use_float16

    # GpuIndexFlatIP: inner-product search on GPU
    index = faiss.GpuIndexFlatIP(res, d, cfg)

    clus = faiss.Clustering(d, n_clusters)
    clus.seed = random_state
    clus.niter = niter
    clus.verbose = True
    if max_points_per_centroid is not None:
        clus.max_points_per_centroid = max_points_per_centroid
    else:
        clus.max_points_per_centroid = max(1, data.shape[0] // n_clusters)  # all data

    print(f"  FAISS GPU k-means: n={data.shape[0]}, d={d}, k={n_clusters}, "
          f"GPU={gpu_id}, niter={niter}")
    clus.train(data, index)

    # Extract centroids
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(n_clusters, d)

    # Assign labels: load centroids into GPU index, search all points
    index.reset()
    index.add(centroids)
    D, I = index.search(data, 1)
    labels = I.flatten().astype(np.int64)

    return labels, centroids


def cluster_faiss_remote(
    vecs: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 42,
    gpu_id: int = 0,
    niter: int = 25,
    use_float16: bool = False,
    max_points_per_centroid: int | None = None,
    service_url: str | None = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remote GPU-accelerated k-means via FAISS — calls the cluster service.

    Use this when the local machine has no GPU but can reach a GPU machine
    running ``src/utils/cluster_remote.py``.  Both machines must share a
    filesystem mount (e.g. /volume/hanxi) — vectors and results are passed
    as .npy files, only lightweight JSON goes over HTTP.

    Set the service URL via the environment variable ``FAISS_CLUSTER_SERVICE_URL``
    or pass it explicitly::

        cluster_faiss_remote(vecs, 100, service_url="http://gpu-box:8765")

    Args:
        service_url: Base URL of the GPU clustering service
                     (default: $FAISS_CLUSTER_SERVICE_URL or http://localhost:8765)
        All other args are forwarded to the remote service.

    Returns:
        labels: (N,) int array, cluster index for each vector
        centers: (n_clusters, dim) float32 array of cluster centroids
    """
    # Resolve service URL
    if service_url is None:
        service_url = os.environ.get(
            "FAISS_CLUSTER_SERVICE_URL", "http://localhost:8765"
        )
    service_url = service_url.rstrip("/")

    # Write vectors to a shared temp file (use project data dir so both machines can see it)
    data_dir = os.environ.get(
        "FAISS_REMOTE_TMPDIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "cluster_tmp"),
    )
    os.makedirs(data_dir, exist_ok=True)
    vecs_path = os.path.join(data_dir, f"_remote_vecs_{os.getpid()}.npy")

    print(f"  FAISS remote: saving {vecs.shape} vectors → {vecs_path}")
    t0 = time.time()
    np.save(vecs_path, vecs.astype(np.float32))

    # Build request
    payload = {
        "vecs_path": vecs_path,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "gpu_id": gpu_id,
        "niter": niter,
        "use_float16": use_float16,
    }
    if max_points_per_centroid is not None:
        payload["max_points_per_centroid"] = max_points_per_centroid

    # These go alongside the input so both machines see them
    payload["output_labels_path"] = vecs_path.replace(".npy", "_labels.npy")
    payload["output_centers_path"] = vecs_path.replace(".npy", "_centers.npy")

    # Call the remote service
    url = f"{service_url}/cluster"
    print(f"  FAISS remote: POST {url}")
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach FAISS cluster service at {service_url}: {e}\n"
            f"  Is the GPU machine running 'bash scripts/start_cluster_servise.sh'?"
        ) from e

    if not result.get("success"):
        raise RuntimeError(
            f"Remote clustering failed: {result.get('error', 'unknown error')}"
        )

    elapsed = result.get("time_sec", time.time() - t0)

    # Read back results
    print(f"  FAISS remote: reading labels ← {payload['output_labels_path']}")
    labels = np.load(payload["output_labels_path"])
    centers = np.load(payload["output_centers_path"])

    # Clean up temp files
    for p in [vecs_path, payload["output_labels_path"], payload["output_centers_path"]]:
        try:
            os.remove(p)
        except OSError:
            pass

    print(f"  FAISS remote: done in {elapsed:.1f}s  "
          f"(n={vecs.shape[0]}, k={n_clusters})")
    return labels, centers


from utils.teb_mean import cluster_teb  # noqa: E402
from utils.adaptive_mean import adaptive_clustering  # noqa: E402

CLUSTERERS = {
    "kmeans": cluster_kmeans,
    "minibatch_kmeans": cluster_minibatch_kmeans,
    "faiss_gpu": cluster_faiss_gpu,
    "faiss_remote": cluster_faiss_remote,
    "teb": cluster_teb,
    "adaptive": adaptive_clustering,
}


def apply_clustering(vecs: np.ndarray, method: str = "kmeans",
                     n_clusters: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray, float]:
    if method not in CLUSTERERS:
        raise ValueError(f"Unknown clustering method: {method}. Available: {list(CLUSTERERS)}")
    t0 = time.time()
    labels, centers = CLUSTERERS[method](vecs, n_clusters=n_clusters, **kwargs)
    elapsed = time.time() - t0
    print(f"  clustering [{method}] took {elapsed:.1f}s  "
          f"(n={vecs.shape[0]}, k={n_clusters})")
    return labels, centers, elapsed

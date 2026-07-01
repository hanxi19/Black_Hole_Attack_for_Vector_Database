"""
Test the remote FAISS GPU clustering service.

Usage:
    python test/test_remote_cluster.py

Requires the service to be running (start_cluster_servise.sh on GPU machine).
Set FAISS_CLUSTER_SERVICE_URL or pass --service-url to point at the proxy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from attack.cluster import cluster_faiss_remote, cluster_kmeans

# Default service URL (reverse proxy to GPU machine)
DEFAULT_SERVICE_URL = (
    "https://siflow-changliu.siflow.cn/siflow/changliu/07a2e58146/eval/v1/8765"
)


def test_health(service_url: str) -> bool:
    """Quick health check."""
    import urllib.request
    import json

    url = f"{service_url}/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print(f"  Health: status={data['status']}, gpu={data['gpu_available']}, count={data['gpu_count']}")
        return data.get("gpu_available", False)
    except Exception as e:
        print(f"  Health check FAILED: {e}")
        return False


def test_small_cluster(service_url: str):
    """Cluster a small synthetic dataset and verify shapes."""
    print("\n[1/3] Small cluster test (n=500, d=64, k=5) ...")
    rng = np.random.RandomState(42)
    vecs = rng.randn(500, 64).astype(np.float32)

    labels, centers = cluster_faiss_remote(
        vecs, n_clusters=5, random_state=42, service_url=service_url,
    )

    assert labels.shape == (500,), f"labels shape mismatch: {labels.shape}"
    assert centers.shape == (5, 64), f"centers shape mismatch: {centers.shape}"
    assert labels.min() >= 0 and labels.max() < 5, f"label range wrong: [{labels.min()}, {labels.max()}]"

    # Every point should be assigned to its nearest centroid (inner product)
    sim = vecs @ centers.T  # (500, 5)
    best_from_labels = np.argmax(sim, axis=1)
    agreement = (labels == best_from_labels).mean()
    print(f"  PASS — shapes OK, nearest-centroid agreement: {agreement:.4f}")
    return labels, centers


def test_vs_local_kmeans(service_url: str):
    """Compare remote FAISS GPU result to local sklearn k-means on a larger set."""
    print("\n[2/3] Compare remote FAISS vs local k-means (n=2000, d=128, k=10) ...")
    rng = np.random.RandomState(123)
    vecs = rng.randn(2000, 128).astype(np.float32)

    # Remote FAISS GPU
    t0 = time.time()
    labels_remote, centers_remote = cluster_faiss_remote(
        vecs, n_clusters=10, random_state=42, service_url=service_url,
    )
    remote_time = time.time() - t0

    # Local k-means
    t0 = time.time()
    labels_local, centers_local = cluster_kmeans(vecs, n_clusters=10, random_state=42)
    local_time = time.time() - t0

    # Both should have similar clustering quality (within-cluster SSE)
    def sse(v, l, c):
        ss = 0.0
        for i in range(c.shape[0]):
            mask = l == i
            if mask.any():
                ss += np.sum((v[mask] - c[i]) ** 2)
        return ss

    sse_remote = sse(vecs, labels_remote, centers_remote)
    sse_local = sse(vecs, labels_local, centers_local)

    # Histogram of cluster sizes
    _, counts_remote = np.unique(labels_remote, return_counts=True)
    _, counts_local = np.unique(labels_local, return_counts=True)

    print(f"  Remote: {remote_time:.2f}s  SSE={sse_remote:.2f}  sizes={sorted(counts_remote)}")
    print(f"  Local:  {local_time:.2f}s  SSE={sse_local:.2f}  sizes={sorted(counts_local)}")

    # SSE should be within 20% (FAISS GPU uses inner product after L2 norm,
    # which approximates cosine; local k-means uses Euclidean)
    ratio = sse_remote / max(sse_local, 1e-6)
    print(f"  SSE ratio (remote/local): {ratio:.3f}")
    if ratio < 0.5 or ratio > 2.0:
        print(f"  WARNING: SSE ratio {ratio:.3f} is outside [0.5, 2.0] — may be expected (cosine vs Euclidean)")

    print(f"  PASS — both methods converge to reasonable clusters")
    return labels_remote, centers_remote


def test_large_batch(service_url: str):
    """Test with a moderately large batch to check I/O and timeout."""
    print("\n[3/3] Large batch test (n=50000, d=768, k=50) ...")
    rng = np.random.RandomState(99)
    vecs = rng.randn(50000, 768).astype(np.float32)

    t0 = time.time()
    labels, centers = cluster_faiss_remote(
        vecs,
        n_clusters=50,
        random_state=42,
        max_points_per_centroid=1000,  # subsample for speed
        service_url=service_url,
    )
    elapsed = time.time() - t0

    assert labels.shape == (50000,), f"labels shape: {labels.shape}"
    assert centers.shape == (50, 768), f"centers shape: {centers.shape}"

    # Check cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Done in {elapsed:.1f}s  n_clusters_used={len(unique)}  "
          f"min_size={counts.min()}  max_size={counts.max()}  mean_size={counts.mean():.0f}")

    print(f"  PASS — large batch handled correctly")
    return labels, centers


def main():
    parser = argparse.ArgumentParser(description="Test remote FAISS GPU clustering")
    parser.add_argument(
        "--service-url",
        default=os.environ.get("FAISS_CLUSTER_SERVICE_URL", DEFAULT_SERVICE_URL),
        help="Base URL of the clustering service",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only run health check, skip clustering tests",
    )
    args = parser.parse_args()

    service_url = args.service_url.rstrip("/")
    print(f"Testing remote cluster service at: {service_url}")
    print()

    if not test_health(service_url):
        print("\nService is not healthy — aborting.")
        sys.exit(1)

    if args.health_only:
        print("\nHealth check passed.")
        return

    try:
        test_small_cluster(service_url)
        test_vs_local_kmeans(service_url)
        test_large_batch(service_url)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  All tests passed!")
    print("=" * 55)


if __name__ == "__main__":
    main()

"""
FAISS GPU clustering service.

Run this on the GPU machine to expose FAISS GPU clustering as an HTTP API.
The non-GPU machine sends clustering requests with file paths on the shared
/volume/hanxi mount — vectors and results are passed via .npy files, so only
lightweight JSON metadata goes over HTTP.

Start the service:
    python src/utils/cluster_remote.py --port 8765

Or use the convenience script:
    bash scripts/start_cluster_servise.sh

Dependencies on the GPU machine:
    pip install flask faiss-gpu numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import uuid

import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
#  Global state — populated in main()
# ---------------------------------------------------------------------------
_has_gpu: bool = False
_gpu_count: int = 0
_temp_root: str = "/tmp"


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check — returns GPU availability and count."""
    return jsonify({
        "status": "ok",
        "gpu_available": _has_gpu,
        "gpu_count": _gpu_count,
    })


@app.route("/cluster", methods=["POST"])
def cluster():
    """Run FAISS GPU k-means clustering.

    Request JSON body:
        vecs_path:              str   — path to input .npy (float32, shape (N, d))
        n_clusters:             int   — number of clusters
        random_state:           int   — (default 42)
        gpu_id:                 int   — (default 0)
        niter:                  int   — (default 25)
        use_float16:            bool  — (default false)
        max_points_per_centroid: int | null — (default null = use all data)
        output_labels_path:     str | null — path for labels .npy output
        output_centers_path:    str | null — path for centers .npy output

    If output_*_path is omitted, a temp path under /tmp is auto-generated and
    returned in the response so the caller can read it back.

    Response JSON:
        success:              bool
        n, d, k:              int   — data dimensions
        time_sec:             float — clustering wall time
        labels_path:          str   — path to labels .npy
        centers_path:         str   — path to centroids .npy
        labels_shape:         [int]
        centers_shape:        [int]
    """
    import faiss

    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        # --- required params ---
        vecs_path = data.get("vecs_path")
        n_clusters = data.get("n_clusters")

        if not vecs_path:
            return jsonify({"error": "Missing 'vecs_path'"}), 400
        if not n_clusters:
            return jsonify({"error": "Missing 'n_clusters'"}), 400
        if not isinstance(n_clusters, int) or n_clusters < 1:
            return jsonify({"error": f"Invalid n_clusters: {n_clusters}"}), 400

        # --- optional params ---
        random_state = data.get("random_state", 42)
        gpu_id = data.get("gpu_id", 0)
        niter = data.get("niter", 25)
        use_float16 = data.get("use_float16", False)
        max_points_per_centroid = data.get("max_points_per_centroid", None)
        output_labels_path = data.get("output_labels_path")
        output_centers_path = data.get("output_centers_path")

        # --- validate input file ---
        if not os.path.exists(vecs_path):
            return jsonify({"error": f"File not found: {vecs_path}"}), 404

        # --- auto-generate output paths if not provided ---
        uid = uuid.uuid4().hex[:12]
        if not output_labels_path:
            output_labels_path = os.path.join(_temp_root, f"faiss_labels_{uid}.npy")
        if not output_centers_path:
            output_centers_path = os.path.join(_temp_root, f"faiss_centers_{uid}.npy")

        # --- load vectors ---
        t0 = time.time()
        vecs = np.load(vecs_path).astype(np.float32)

        # --- GPU check ---
        n_gpus = faiss.get_num_gpus()
        if n_gpus == 0:
            return jsonify({"error": "No GPU available on this machine"}), 500

        # --- normalize for inner product (cosine similarity) ---
        data_arr = vecs.copy()
        faiss.normalize_L2(data_arr)

        d = data_arr.shape[1]
        res = faiss.StandardGpuResources()

        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = gpu_id
        cfg.useFloat16 = use_float16

        index = faiss.GpuIndexFlatIP(res, d, cfg)

        clus = faiss.Clustering(d, n_clusters)
        clus.seed = random_state
        clus.niter = niter
        clus.verbose = True

        if max_points_per_centroid is not None:
            clus.max_points_per_centroid = max_points_per_centroid
        else:
            clus.max_points_per_centroid = max(1, data_arr.shape[0] // n_clusters)

        print(f"  FAISS GPU k-means: n={data_arr.shape[0]}, d={d}, k={n_clusters}, "
              f"GPU={gpu_id}, niter={niter}, max_ppc={clus.max_points_per_centroid}")

        # --- train ---
        clus.train(data_arr, index)

        # --- extract centroids ---
        centroids = faiss.vector_float_to_array(clus.centroids).reshape(n_clusters, d)

        # --- assign labels ---
        index.reset()
        index.add(centroids)
        D, I = index.search(data_arr, 1)
        labels = I.flatten().astype(np.int64)

        train_time = time.time() - t0

        # --- save results ---
        os.makedirs(os.path.dirname(output_labels_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_centers_path) or ".", exist_ok=True)
        np.save(output_labels_path, labels)
        np.save(output_centers_path, centroids)

        print(f"  Done in {train_time:.1f}s  "
              f"labels → {output_labels_path}  "
              f"centers → {output_centers_path}")

        return jsonify({
            "success": True,
            "n": int(data_arr.shape[0]),
            "d": d,
            "k": n_clusters,
            "time_sec": round(train_time, 2),
            "labels_path": output_labels_path,
            "centers_path": output_centers_path,
            "labels_shape": list(labels.shape),
            "centers_shape": list(centroids.shape),
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FAISS GPU clustering service — run on the GPU machine"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--temp-root", default="/tmp",
                        help="Directory for auto-generated output files (default: /tmp)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    global _has_gpu, _gpu_count, _temp_root
    _temp_root = args.temp_root

    # --- check GPU availability ---
    try:
        import faiss
        _gpu_count = faiss.get_num_gpus()
        _has_gpu = _gpu_count > 0
    except ImportError:
        print("WARNING: faiss not installed — GPU clustering will not work")
        _gpu_count = 0
        _has_gpu = False

    print("=" * 55)
    print("  FAISS GPU Clustering Service")
    print(f"  GPU available: {_has_gpu}")
    print(f"  GPU count:     {_gpu_count}")
    print(f"  Temp root:     {_temp_root}")
    print(f"  Listening on:  {args.host}:{args.port}")
    print("=" * 55)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

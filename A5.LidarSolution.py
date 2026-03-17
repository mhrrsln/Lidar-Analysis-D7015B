from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

here        = Path(__file__).parent
datadir     = here / "LidarData"
resultsdir    = here / "ResultsofA5"
resultsdir.mkdir(parents=True, exist_ok=True)

dataset_paths = {
    "dataset1": datadir / "dataset1.npy",
    "dataset2": datadir / "dataset2.npy"
}

# DBSCAN Parameters
min_samples      = 5     # minimum points to form a core point
k_neighbors     = 5     # k used in the k-distance elbow plot
ground_offset   = 1.0   # meters above histogram peak → filter threshold

# Utility Helpers
# ---------------

def show_cloud(points, title="Point Cloud"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.01)
    ax.set_title(title)
    plt.show()

# Task1 - Ground Level Detection
# ------------------------------

def get_ground_level(pcd: np.ndarray, n_bins: int = 200) -> float:
    counts, bin_edges = np.histogram(pcd[:, 2], bins=n_bins)
    peak_bin = np.argmax(counts)
    ground_z = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
    return ground_z

def plot_histogram(pcd: np.ndarray, ground_z: float, threshold: float,
                   name: str) -> None:
    counts, bin_edges = np.histogram(pcd[:, 2], bins=200)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
           align="edge", color="#1C11BB", edgecolor="none", alpha=0.8)
    ax.axvline(ground_z, color="#E2348B", linewidth=2,
               label=f"Ground Peak = {ground_z:.2f}m")
    ax.axvline(threshold, color="#DF791B", linewidth=2, linestyle="--",
               label=f"Cut Threshold = {threshold:.2f}m")
    ax.set_xlabel("Z-Height (m)", fontsize=12)
    ax.set_ylabel("Point Count", fontsize=12)
    ax.set_title(f"{name} - Z Histogram (Ground Detection)", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = resultsdir / f"{name}_histogram.png"
    plt.savefig(path, dpi=600)
    plt.close()
    print(f" [Task1] saved Histogram → {path}")

# Task2 - Optimal EPS via K-Distance Elbow
# ----------------------------------------

def find_optimal_eps(pcd_ag: np.ndarray, k: int = k_neighbors) -> float:
    nbrs = NearestNeighbors(n_neighbors=k).fit(pcd_ag)
    distances, _ = nbrs.kneighbors(pcd_ag)
    k_dist = np.sort(distances[:, -1])[::-1] # decending

    # Max-Curvature Elbow via Perpendicular Distance to the Chord p1→p2
    n           = len(k_dist)
    coords      = np.column_stack([np.arange(n), k_dist])
    chord       = coords[-1] - coords[0]
    chord_len   =np.linalg.norm(chord)
    vecs        = coords - coords[0]
    cross       = np.abs(vecs[:, 0] * chord[1] - vecs[:, 1] * chord[0])
    elbow_idx   = int(np.argmax(cross / chord_len))
    optimal_eps = float(k_dist[elbow_idx])
    return optimal_eps, k_dist, elbow_idx

def plot_elbow(k_dist: np.ndarray, elbow_idx: int, optimal_eps: float,
               name: str) -> None:
    display_n = min(5000, len(k_dist)) # limit x-axis for clarity

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_dist[:display_n], color="#173AD3", linewidth=1.2, label="k-dist Curve")
    ax.axvline(elbow_idx, color="#D61A59", linewidth=2, linestyle="--",
               label=f"Elbow IDX = {elbow_idx}")
    ax.axhline(optimal_eps, color="#E26923", linewidth=2, linestyle="--",
               label=f"Optimal EPS = {optimal_eps:.3f}")
    ax.set_xlabel("Points (Sorted Descending)", fontsize=12)
    ax.set_ylabel(f"{k_neighbors} - NN Distance", fontsize=12)
    ax.set_title(f"{name} - K-Distance Elbow (Optimal EPS = {optimal_eps:.3f})",
                 fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = resultsdir / f"{name}_elbow.png"
    plt.savefig(path, dpi=600)
    plt.close()
    print(f" [Task2] saved Elbow Plot → {path}")

def plot_clusters(pcd_ag: np.ndarray, labels: np.ndarray,
                  n_clusters: int, eps: float, name: str) -> None:
    cmap = plt.cm.Spectral(np.linspace(0, 1, max(n_clusters, 1)))

    fig, ax = plt.subplots(figsize=(11, 10))
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(pcd_ag[mask, 0], pcd_ag[mask, 1], s=1,
                       c="#36739C", alpha=0.3, label="Noise")
        else:
            ax.scatter(pcd_ag[mask, 0], pcd_ag[mask, 1], s=2,
                       c=[cmap[lbl % len(cmap)]])
    
    ax.set_title(f"{name} - DBSCAN: {n_clusters} Clusters (EPS = {eps:.3f})",
                 fontsize=14)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    plt.tight_layout()
    path = resultsdir / f"{name}_clusters.png"
    plt.savefig(path, dpi=600)
    plt.close()
    print(f" [Task2] saved Cluster Plot → {path}")

# Task3 - Catenary Cluster Extraction
# -----------------------------------

def find_catenary_cluster(pcd_ag: np.ndarray, labels: np.ndarray):
    unique_labels = [l for l in set(labels) if l != -1]
    best_label, best_span = -1, -1.0

    for lbl in unique_labels:
        pts = pcd_ag[labels == lbl]
        span = (pts[:, 0].max() - pts[:, 0].min()) + \
               (pts[:, 1].max() - pts[:, 1].min())
        if span > best_span:
            best_span = span
            best_label = lbl

    catenary_pts = pcd_ag[labels == best_label]
    return catenary_pts, best_label

def plot_catenary(catenary: np.ndarray, name: str) -> None:
    minx = catenary[:, 0].min(); maxx = catenary[:, 0].max()
    miny = catenary[:, 1].min(); maxy = catenary[:, 1].max()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(catenary[:, 0], catenary[:, 1], s=2, c="#EE2D8D", alpha=0.7)
    ax.set_title(
        f"{name} - Catenary Cluster\n"
        f"min(x)={minx:.2f} min(y)={miny:.2f} "
        f"max(x)={maxx:.2f} max(y)={maxy:.2f}",
        fontsize=14,
    )
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    plt.tight_layout()
    path = resultsdir / f"{name}_Catenary.png"
    plt.savefig(path, dpi=600)
    plt.close()
    print(f" [Task3] saved Catenary Plot → {path}")

# Pipeline
# -------------

def process_dataset(name: str, path: str) -> dict:
    print(f" Processing {name}")

    # Load
    pcd = np.load(path)
    print(f" Loaded {pcd.shape[0]:,} points "
          f" (Z Range {pcd[:,2].min():.2f} - {pcd[:,2].max():.2f} m)")
    
    # Task 1 
    ground_z = get_ground_level(pcd)
    threshold = ground_z + ground_offset
    print(f"\n [Task1] Ground Peak = {ground_z:.3f} m")
    print(f" [Task1] Threshold = {threshold:.3f} m (Peak + {ground_offset} m")
    plot_histogram(pcd, ground_z, threshold, name)

    pcd_ag = pcd[pcd[:, 2] > threshold]
    print(f" [Task1] Points above Ground: {pcd_ag.shape[0]:,}")

    # Task2
    optimal_eps, k_dist, elbow_idx = find_optimal_eps(pcd_ag)
    print(f"\n [Task2] Optimal EPS = {optimal_eps:.4f}")
    plot_elbow(k_dist, elbow_idx, optimal_eps, name)

    clustering  = DBSCAN(eps=optimal_eps, min_samples=min_samples).fit(pcd_ag)
    labels      = clustering.labels_
    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    print(f" [Task2] Clusters found: {n_clusters}")
    plot_clusters(pcd_ag, labels, n_clusters, optimal_eps, name)

    # Task3
    catenary, cat_lbl = find_catenary_cluster(pcd_ag, labels)
    minx = catenary[:, 0].min(); maxx = catenary[:, 0].max()
    miny = catenary[:, 1].min(); maxy = catenary[:, 1].max()
    print(f"\n [Task3] Catenary Label : {cat_lbl} ({len(catenary):,} points)")
    print(f" [Task3] min(x) = {minx:.3f} | max(x) = {maxx:.3f}")
    print(f" [Task3] min(y) = {miny:.3f} | max(y) = {maxy:.3f}")
    plot_catenary(catenary, name)

    return {
        "ground_level"      : round(ground_z, 3),
        "ground_threshold"  : round(threshold, 3),
        "optimal_eps"       : round(optimal_eps, 4),
        "n_clusters"        : n_clusters,
        "catenary_points"   : len(catenary),
        "catenary_minx"     : round(minx, 3),
        "catenary_miny"     : round(miny, 3),
        "catenary_maxx"     : round(maxx, 3),
        "catenary_maxy"     : round(maxy, 3),
    }

def generate_readme(all_results: dict) -> None:

    img_dir = "ResultsofA5"

    lines = []
    lines.append("Assignment 5 - LIDAR Point Cloud Analysis\n\n")
    lines.append("D7015B - Industrial AI and eMaintenance - Part I Theories and Concepts \n")
    lines.append("Datasets : Aerial LIDAR Scan of a Railway Track Section\n\n\n")

    # Task1
    lines.append("Task1 - Ground Level Detection\n\n")
    lines.append(
        "The ground plane is detected by finding the peak (mode) of the z-histogram. "
        "The tallest bar represent the flat terrain which contains the most points. "
        "A threshold of **peak + 1m** is applied to remove ground points. \n\n" 
    )
    for name, r in all_results.items():
        lines.append(f" {name}\n\n")
        lines.append(f" | Metric           | Value          |\n")
        lines.append(f" |------------------|----------------|\n")
        lines.append(f" | Ground Level (m) | {r['ground_level']:<12} |\n")
        lines.append(f" | Threshold (m)    | {r['ground_threshold']:<12} |\n\n")
        lines.append(f" ![{name} Histogram]({img_dir}/{name}_histogram.png)\n\n\n")
    
    # Task2
    lines.append("Task2 - Optimal DBSCAN EPS (Elbow Method)\n\n")
    lines.append(
        "The optimal epsilon is found by computing the 5th nearest-neighbour distance "
        "for every point, sorting descending, and locating the maximum-curvature elbow "
        "of the resulting curve. That distance value is used as the DBSCAN EPS parameter.\n\n"
    )

    for name, r in all_results.items():
        lines.append(f" {name}\n\n")
        lines.append(f" | Metric           | Value          |\n")
        lines.append(f" |------------------|----------------|\n")
        lines.append(f" | Optimal EPS      | {r['optimal_eps']:<12} |\n")
        lines.append(f" | Clusters found   | {r['n_clusters']:<12} |\n\n")
        lines.append(f" ![{name} Elbow]({img_dir}/{name}_elbow.png)\n")
        lines.append(f" ![{name} Clusters]({img_dir}/{name}_clusters.png)\n\n\n")
    
    # Task3
    lines.append("Task3 - Catenary Cluster Extraction\n\n")
    lines.append(
        "The catenary (over power wire) cluster is identified as the cluster with "
        "the largest combined XY span (x_range + y_range), since the wire runs along "
        "the entire length of the scanned track.\n\n"
    )

    for name, r in all_results.items():
        lines.append(f" {name}\n\n")
        lines.append(f" | Metric | Value          |\n")
        lines.append(f" |--------|----------------|\n")
        lines.append(f" | min(x) | {r['catenary_minx']:<12} |\n")
        lines.append(f" | min(y) | {r['catenary_miny']:<12} |\n")
        lines.append(f" | max(x) | {r['catenary_maxx']:<12} |\n")
        lines.append(f" | max(y) | {r['catenary_maxy']:<12} |\n\n")
        lines.append(f" ![{name} Catenary]({img_dir}/{name}_catenary.png)\n\n\n")
    
    lines.append("Genarated automatically by 'A5LidarSolution.py'\n")

    readme_path = here / "README.md"
    readme_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n README.md saved → {readme_path}")

# Main
# ----

if __name__ == "__main__":
    all_results = {}
    for ds_name, ds_path in dataset_paths.items():
        all_results[ds_name] = process_dataset(ds_name, ds_path)
    generate_readme(all_results)
#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
from PIL import Image

THUMB_SIZE = (48, 32)  # fixed small size so comparisons are fast and size-invariant
TIME_BUDGET_SECONDS = 240  # total wall-clock budget, measured from program start

# Above this many frames, the O(n^2) full distance matrix + 2-opt pipeline
# gets too slow to finish in the time budget, so we switch to a cheaper
# KD-tree-based greedy construction instead (see fast_greedy_path).
MAX_FULL_MATRIX_FRAMES = 1500


def load_frames(frame_dir):
    names = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(".png"))
    vectors = []
    for name in names:
        img = Image.open(os.path.join(frame_dir, name)).convert("L")
        img = img.resize(THUMB_SIZE, Image.BILINEAR)
        vectors.append(np.asarray(img, dtype=np.float32).ravel())
    return names, np.stack(vectors)


def pairwise_distances(vectors):
    n = vectors.shape[0]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dist[i] = np.abs(vectors[i] - vectors).mean(axis=1)
    return dist


def nearest_neighbor_path(dist, start):
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    path = [start]
    visited[start] = True
    for _ in range(n - 1):
        d = dist[path[-1]].copy()
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        path.append(nxt)
        visited[nxt] = True
    return path


def path_length(dist, path):
    return float(sum(dist[path[i], path[i + 1]] for i in range(len(path) - 1)))


def fast_greedy_path(vectors, deadline):
    """Greedy nearest-neighbor chain for very large frame counts.

    Avoids ever materializing an O(n^2) distance matrix: uses a KD-tree
    for nearest-neighbor queries, with lazy deletion (a visited mask plus
    periodic tree rebuilds) so it stays close to O(n log n) overall
    instead of O(n^2). No 2-opt refinement is attempted at this scale;
    a single greedy pass is the best we can afford in the time budget.
    """
    from scipy.spatial import cKDTree

    n = vectors.shape[0]
    visited = np.zeros(n, dtype=bool)
    remaining = np.arange(n)
    tree = cKDTree(vectors[remaining])

    current = 0
    path = [current]
    visited[current] = True
    stale = 1  # visited frames counted against the current tree

    for _ in range(n - 1):
        if time.time() > deadline:
            # Out of time: append whatever is left in arbitrary order
            # rather than returning a partial/invalid path.
            leftovers = np.where(~visited)[0]
            path.extend(int(i) for i in leftovers)
            break

        if stale > max(8, len(remaining) // 5):
            remaining = np.where(~visited)[0]
            tree = cKDTree(vectors[remaining])
            stale = 0

        k = 2
        found = None
        while found is None:
            k = min(k, len(remaining))
            _, idxs = tree.query(vectors[current], k=k)
            idxs = np.atleast_1d(idxs)
            for local_idx in idxs:
                candidate = int(remaining[local_idx])
                if not visited[candidate]:
                    found = candidate
                    break
            if found is None:
                if k >= len(remaining):
                    # Tree is fully stale; rebuild from scratch and retry.
                    remaining = np.where(~visited)[0]
                    tree = cKDTree(vectors[remaining])
                    stale = 0
                    k = 2
                else:
                    k *= 2

        path.append(found)
        visited[found] = True
        current = found
        stale += 1

    return path


def two_opt(dist, path, deadline):
    n = len(path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 2):
            if time.time() > deadline:
                break
            a, b = path[i], path[i + 1]
            js = np.arange(i + 2, n)
            if len(js) == 0:
                continue
            delta = np.full(len(js), np.inf, dtype=np.float32)
            has_next = js < n - 1
            jn = js[has_next]
            if len(jn) > 0:
                cn = np.array([path[j] for j in jn])
                dnext = np.array([path[j + 1] for j in jn])
                delta[has_next] = (
                    dist[a, cn] + dist[b, dnext] - dist[a, b] - dist[cn, dnext]
                )
            best_idx = int(np.argmin(delta))
            if delta[best_idx] < -1e-6:
                j = int(js[best_idx])
                path[i + 1 : j + 1] = path[i + 1 : j + 1][::-1]
                improved = True
    return path


def main():
    start_time = time.time()
    deadline = start_time + TIME_BUDGET_SECONDS

    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "/frames"
    names, vectors = load_frames(frame_dir)
    n = len(names)
    if n <= 2:
        for name in names:
            print(name)
        return

    if n > MAX_FULL_MATRIX_FRAMES:
        # Too many frames for an O(n^2) matrix + 2-opt within the time
        # budget: fall back to a cheaper KD-tree greedy construction.
        best_path = fast_greedy_path(vectors, deadline)
    else:
        dist = pairwise_distances(vectors)

        best_path, best_len = None, None
        starts = range(n) if n <= 20 else [0, n // 2, n - 1]
        for s in starts:
            p = nearest_neighbor_path(dist, s)
            length = path_length(dist, p)
            if best_len is None or length < best_len:
                best_path, best_len = p, length

        best_path = two_opt(dist, best_path, deadline)

    for idx in best_path:
        print(names[idx])


if __name__ == "__main__":
    main()

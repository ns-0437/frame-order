#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
from PIL import Image

THUMB_SIZE = (48, 32)  # fixed small size so comparisons are fast and size-invariant
TIME_BUDGET_SECONDS = 240  # total wall-clock budget, measured from program start


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

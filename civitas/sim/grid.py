"""Small numpy grid helpers (no scipy dependency)."""
from __future__ import annotations

from collections import deque

import numpy as np


def shift(a: np.ndarray, dy: int, dx: int, fill=0) -> np.ndarray:
    """Return `a` shifted by (dy, dx); vacated cells get `fill`."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Chebyshev dilation of a boolean mask by radius r."""
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    # separable: dilate rows then columns
    tmp = mask.copy()
    for d in range(1, r + 1):
        tmp |= shift(mask, 0, d, False) | shift(mask, 0, -d, False)
    out = tmp.copy()
    for d in range(1, r + 1):
        out |= shift(tmp, d, 0, False) | shift(tmp, -d, 0, False)
    return out


def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)x(2r+1) square neighbourhood (zero padded)."""
    a = a.astype(float)
    p = np.pad(a, r)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    h, w = a.shape
    return c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]


def make_kernel(radius: int, falloff: float) -> list[tuple[int, int, float]]:
    ker = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = (dx * dx + dy * dy) ** 0.5
            if d <= radius + 0.5:
                ker.append((dy, dx, float(np.exp(-d / falloff))))
    return ker


def convolve(src: np.ndarray, kernel: list[tuple[int, int, float]]) -> np.ndarray:
    out = np.zeros_like(src, dtype=float)
    if not src.any():
        return out
    for dy, dx, w in kernel:
        out += w * shift(src, dy, dx, 0.0)
    return out


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labelling. Returns (labels, n); -1 = background."""
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if labels[y0, x0] != -1:
            continue
        q = deque([(y0, x0)])
        labels[y0, x0] = n
        while q:
            y, x = q.popleft()
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == -1:
                    labels[ny, nx] = n
                    q.append((ny, nx))
        n += 1
    return labels, n


INF = 1 << 20
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def road_bfs(road: np.ndarray, seeds: list[tuple[int, int]], max_d: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Multi-source BFS along road tiles (4-connected), vectorised with numpy.

    Each seed is a facility tile (x, y); it enters the road network through
    the road tiles in its 8-neighbourhood (distance 1). Returns
    (dist, label): road-step distance to the nearest seed and that seed's
    index (-1 / INF where unreachable or not a road).
    """
    h, w = road.shape
    dist = np.full((h, w), INF, np.int32)
    label = np.full((h, w), -1, np.int32)
    for k, seed in enumerate(seeds):
        tiles = seed if isinstance(seed, list) else [seed]     # a multi-tile building enters from any part
        for (x, y), dy, dx in ((t, dy, dx) for t in tiles for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and road[ny, nx] and dist[ny, nx] > 1:
                dist[ny, nx] = 1
                label[ny, nx] = k
    frontier = dist == 1
    step = 1
    while frontier.any() and step < max_d:
        step += 1
        new = np.zeros_like(frontier)
        for dy, dx in _DIRS4:
            m = shift(frontier, dy, dx, False) & road & (dist == INF) & ~new
            if m.any():
                dist[m] = step
                label[m] = shift(label, dy, dx, -1)[m]
                new |= m
        frontier = new
    return dist, label


def spread_to_tiles(dist: np.ndarray, label: np.ndarray, r: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Extend a road distance field to the tiles within Chebyshev `r` of a road."""
    td, tl = dist.copy(), label.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if not (dx or dy):
                continue
            sd = shift(dist, dy, dx, INF) + max(abs(dx), abs(dy))
            m = sd < td
            if m.any():
                td[m] = sd[m]
                tl[m] = shift(label, dy, dx, -1)[m]
    return td, tl


def distance_to(mask: np.ndarray) -> np.ndarray:
    """4-connected BFS distance from any True cell (inf-like 999 if none)."""
    h, w = mask.shape
    dist = np.full((h, w), 999, dtype=np.int32)
    q = deque()
    for y, x in zip(*np.nonzero(mask)):
        dist[y, x] = 0
        q.append((int(y), int(x)))
    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] > d:
                dist[ny, nx] = d
                q.append((ny, nx))
    return dist

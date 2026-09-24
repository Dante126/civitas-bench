"""Parse the ASCII map in an observation back into arrays (for scripted agents)."""
from __future__ import annotations

from collections import deque

import numpy as np

from ..sim.grid import dilate

ROADISH = set("#=H")


class MapView:
    def __init__(self, ascii_map: str):
        rows = [ln[4:] for ln in ascii_map.splitlines()[2:]]
        self.h, self.w = len(rows), len(rows[0])
        self.g = np.array([list(r) for r in rows])

    def find(self, ch: str) -> list[tuple[int, int]]:
        ys, xs = np.nonzero(self.g == ch)
        return list(zip(xs.tolist(), ys.tolist()))

    def mask(self, chars: str) -> np.ndarray:
        return np.isin(self.g, list(chars))

    def connected_roads(self) -> np.ndarray:
        road = self.mask("#=H")
        out = np.zeros_like(road)
        q = deque(self.find("H"))
        for x, y in q:
            out[y, x] = True
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < self.w and 0 <= ny < self.h and road[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    q.append((nx, ny))
        return out

    def access(self, radius: int = 2) -> np.ndarray:
        return dilate(self.connected_roads(), radius)

    def river_span(self, y: int) -> tuple[int, int] | None:
        xs = np.nonzero(np.isin(self.g[y], ["~", "="]))[0]
        return (int(xs.min()), int(xs.max())) if len(xs) else None

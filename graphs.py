from collections import deque
from typing import Iterable

import numpy as np
from tqdm import tqdm

# A concept used in these:
#   [...] get an adjacent edge of the tail whose end does not occur in the open path and
#   the order of its end is greater than the order of the head.
# This ensures that there are no b-loops and that we never emit the same loop twice:
# only the one with the leftmost start gets processed


def enumerate_cycles_liu(adj: np.ndarray, kmax=None):
    # http://btn1x4.inf.uni-bayreuth.de/publications/dotor_buchmann/GUI-Generation/Liu2006%20-%20A%20new%20way%20to%20enumerate%20cycles%20in%20graph.pdf
    if kmax is None:
        kmax = 0xFFFFFFFF

    assert len(adj.shape) == 2, "Adjacency matrix must be 2D"
    assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square"

    vcount = adj.shape[0]
    indexarr = np.array(range(vcount), dtype=int)
    que = deque()

    for n in indexarr:
        que.append((n,))

    while len(que):
        P: tuple = que.popleft()
        k = len(P)
        vh = P[0]
        vt = P[-1]
        # is this a cycle of at least 3 edges? (A-B-A)
        if (k > 1) and adj[vt, vh]:
            yield P
        # can this path be extended?
        if k < kmax:
            # next edges for tail
            vna = indexarr[adj[vt]]
            for vn in vna:
                if vn > vh and vn not in P:
                    que.append((*P, vn))


def enumerate_fixed_len_cycles(adj: np.ndarray, k):
    pattern = [0] * k
    vcount = adj.shape[0]
    indexarr = np.array(range(vcount), dtype=int)

    def rotate(i: int):
        nonlocal pattern
        vh = pattern[0]
        before = pattern[:i]
        vt = pattern[i-1]
        # what can come after vt?
        ok_for_next = np.isin(indexarr, before, invert=True) & (indexarr > vh)
        vna = indexarr[adj[vt] & ok_for_next]
        if i == k - 1:
            # final digit, don't bother storing in pattern
            for v in vna:
                if adj[v, vh]:
                    yield (*before, v)
        else:
            for v in vna:
                pattern[i] = v
                yield from rotate(i + 1)

    def root():
        nonlocal pattern
        for vh in tqdm(indexarr, miniters=1):
            pattern[0] = vh
            yield from rotate(1)

    if k < 1:
        return
    yield from root()

from collections import deque
from typing import Tuple, Iterable, Optional, List

import numpy as np
from tqdm import tqdm

FOR_PROFILER = False


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
    # can only be a chain part if it connects in *and* out
    any_chain_part = adj.any(axis=1) & adj.any(axis=0)

    def rotate(i: int, vt: int, ok_for_next: np.ndarray):
        nonlocal pattern
        # what can come after vt?
        vna = indexarr[ok_for_next & adj[vt]]
        if i == k - 1:
            vh = pattern[0]
            # final digit, directly check if it heads home
            connects = adj[vna, vh]
            yield from ((*pattern[:i], v) for v in vna[connects])
        else:
            lessparts = ok_for_next.copy()
            for v in vna:
                pattern[i] = v
                lessparts[v] = False
                yield from rotate(i + 1, v, lessparts)
                lessparts[v] = True

    def root():
        nonlocal pattern
        for vh in tqdm(indexarr[any_chain_part], miniters=1):
            if FOR_PROFILER and (vh >= 360):
                break
            pattern[0] = vh
            yield from rotate(1, vh, any_chain_part & (indexarr > vh))

    if k < 1:
        return
    yield from root()


AdjacenyMatrix = np.ndarray
BoolMask = np.ndarray
GraphChain = Tuple


def validate_adjacency(adj: AdjacenyMatrix):
    if len(adj.shape) != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be a 2D square, is " + str(adj.shape))


def adjacency_argindex(adj: AdjacenyMatrix) -> np.ndarray:
    return np.array(range(adj.shape[0]), dtype=int)


def adjacency_where_can_cycle(adj: AdjacenyMatrix) -> BoolMask:
    return adj.any(axis=1) & adj.any(axis=0)


def enumerate_chains(adj: AdjacenyMatrix, k: int, valid_points: Optional[BoolMask] = None, *,
                     only_cycles=False) -> Iterable[GraphChain]:
    """
    Enumerate all chains of length k that could be part of a cycle.
    This function includes duplicate/clockwise order filtering.

    :param AdjacencyMatrix adj: the graph
    :param int k : length of chains
    :param BoolMask valid_points: (optional) mask of indices to include in chains
    :param only_cycles: (default=False) only return chains that are cycles
    :return: iterable of chain tuples
    """
    indexarr = adjacency_argindex(adj)
    if valid_points is None:
        valid_points = np.full_like(indexarr, True, dtype=bool)
    last_mask = np.full_like(indexarr, True, dtype=bool)
    last_loc = k - 1

    # generate the rest of the chain in work[loc:], only using nodes from mask
    def chaingen1(loc: int, work: List[int], mask: np.ndarray):
        vh = work[loc - 1]
        if loc < last_loc:
            # all but last element don't need to check cycles
            vna = indexarr[mask & adj[vh]]
            for v in vna:
                mask[v] = False
                work[loc] = v
                yield from chaingen1(loc + 1, work, mask)
                mask[v] = True
        else:
            # last element doens't need to track duplicates
            vna = indexarr[mask & adj[vh] & last_mask]
            for v in vna:
                work[loc] = v
                yield work

    # first step, does setup and GUI
    def chaingen(mask):
        nonlocal last_mask
        work = [0] * k
        for vh in tqdm(indexarr[mask], miniters=1):
            if only_cycles:
                last_mask = adj.T[vh]
            work[0] = vh
            yield from chaingen1(1, work, mask & (indexarr > vh))

    yield from chaingen(valid_points)

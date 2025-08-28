# umc.py
import heapq


def unique_mapping_clustering(G, t):
    """
    Args:
        G (tuple): (V1, V2, E)
        t (float): limiar de similaridade

    Returns:
        tuple: (clusters, unmatched_V1, unmatched_V2)
    """
    V1, V2, E = G
    C = []
    M1 = set()
    M2 = set()
    Q = []

    for (vi, vj, sim) in E:
        if sim >= t:
            heapq.heappush(Q, (-sim, vi, vj))

    while Q:
        sim, vi, vj = heapq.heappop(Q)
        sim = -sim
        if vi not in M1 and vj not in M2:
            C.append((vi, vj, sim))
            M1.add(vi)
            M2.add(vj)

    unmatched_V1 = V1 - M1
    unmatched_V2 = V2 - M2

    return C, unmatched_V1, unmatched_V2

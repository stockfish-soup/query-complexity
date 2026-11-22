"""Symmetric Boolean evaluation and symmetric hidden-support learning."""
import itertools
import numpy as np
from .pipeline import OracleProblem


def _generators(n):
    for i in range(n-1):
        p = list(range(n))
        p[i], p[i+1] = p[i+1], p[i]
        yield p


def _move(mask, p):
    return sum(((mask >> i) & 1) << p[i] for i in range(len(p)))


def _profile(n, profile):
    if n < 1 or len(profile) != n+1 or not set(profile) <= {0, 1}:
        raise ValueError('A binary profile of length size+1 is required')


def symmetric_evaluation(n, profile):
    """Evaluate h(|x|) with coordinate-bit queries, acted on by S_n."""
    _profile(n, profile)
    inputs = list(range(1 << n))
    answers = np.array([[(x >> i) & 1 for i in range(n)] for x in inputs])
    generators = [([_move(x, p) for x in inputs], p) for p in _generators(n)]
    return OracleProblem(answers, generators, [profile[x.bit_count()] for x in inputs])


def symmetric_junta(n, k, profile):
    """Identify a hidden k-subset from queries S -> h(|A intersect S|)."""
    _profile(k, profile)
    if n < k:
        raise ValueError('Require n >= k')
    concepts = [sum(1 << i for i in A) for A in itertools.combinations(range(n), k)]
    indices = {A:i for i,A in enumerate(concepts)}
    queries = list(range(1 << n))
    answers = np.array([[profile[(A & S).bit_count()] for S in queries] for A in concepts])
    generators = [([indices[_move(A,p)] for A in concepts], [_move(S,p) for S in queries]) for p in _generators(n)]
    return OracleProblem(answers, generators)


def triangle_finding(n):
    """Evaluate triangle existence with edge-bit queries and vertex S_n action."""
    if n < 3:
        raise ValueError('Triangle finding requires at least three vertices')
    edges = list(itertools.combinations(range(n), 2))
    index = {edge:i for i,edge in enumerate(edges)}
    masks = [sum(1 << index[e] for e in itertools.combinations(vertices,2))
             for vertices in itertools.combinations(range(n),3)]
    inputs = range(1 << len(edges))
    answers = np.array([[(x >> i) & 1 for i in range(len(edges))] for x in inputs])
    targets = [int(any(x & mask == mask for mask in masks)) for x in inputs]
    generators = []
    for p in _generators(n):
        edge_action = [index[tuple(sorted((p[a],p[b])))] for a,b in edges]
        generators.append(([_move(x,edge_action) for x in inputs],edge_action))
    return OracleProblem(answers,generators,targets)

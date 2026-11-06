"""Multiscale geodesic area growth on reliable, open Poisson triangle meshes."""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def face_graph(mesh, allowed, face_centers=None):
    n=len(mesh.faces)
    adjacent=np.asarray(mesh.face_adjacency,dtype=np.int64)
    edge=adjacent[allowed[adjacent[:,0]] & allowed[adjacent[:,1]]]
    centers=np.asarray(mesh.triangles_center if face_centers is None else face_centers)
    length=np.linalg.norm(centers[edge[:,0]]-centers[edge[:,1]],axis=1)
    return coo_matrix((np.r_[length,length],(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(n,n)).tocsr()


def growth_curve(graph, sources, exterior, area, scales):
    """Exterior area reached from given face sources at geodesic distances.

    This is a face-center graph approximation, not exact intrinsic geodesics.
    Distances are nonnegative and curves must be monotonic.
    """
    sources=np.asarray(sources,np.int64)
    scales=np.asarray(scales,float)
    if not len(sources):return np.zeros(len(scales)),np.full(len(area),np.inf)
    dist=dijkstra(graph,directed=False,indices=sources,min_only=True,limit=float(scales.max()))
    ids=np.flatnonzero(exterior)
    order=np.argsort(dist[ids])
    d=dist[ids][order]; a=np.asarray(area)[ids][order]
    cumulative=np.cumsum(a)
    end=np.searchsorted(d,scales,side='right')
    result=np.where(end>0,cumulative[np.maximum(0,end-1)],0.0)
    return result,dist


def saturated_growth(curve, tolerance=.08):
    curve=np.asarray(curve,float)
    if len(curve)<4 or curve[-1]<=0:return False
    increments=np.diff(curve)
    return bool(increments[-2:].sum() <= tolerance*curve[-1])

"""Source-consistent largest edge-connected original-global Poisson wall."""
import numpy as np
import trimesh


def source_groups(mesh):
    groups = trimesh.graph.connected_components(mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)), min_len=1)
    groups = sorted(groups, key=len, reverse=True)
    component = np.empty(len(mesh.faces), np.int32)
    for index, rows in enumerate(groups):
        component[rows] = index
    main = np.sort(groups[0])
    vertices, inverse = np.unique(mesh.faces[main], return_inverse=True)
    wall = trimesh.Trimesh(vertices=mesh.vertices[vertices],
                           faces=inverse.reshape(-1, 3), process=False)
    return wall, component, [len(g) for g in groups]

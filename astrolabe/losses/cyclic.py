import itertools
import math
import random

import numpy as np

from astrolabe.losses.base import loss_registry


def calculate_radius_of_gyration(coordinates):
    N = len(coordinates)
    com = np.mean(coordinates, axis=0)
    rg_squared = np.sum((coordinates - com) ** 2) / N
    rg = np.sqrt(rg_squared)
    return rg


def calculate_distance_diff(oligo):
    c = get_coord_CA(oligo)  # get CA atoms
    center_of_mass = _compute_chain_center_of_mass(c)

    # Compare distances between adjacent chains, including first-last.
    chain_order = sorted(center_of_mass.keys())
    next_chain = np.roll(chain_order, -1)
    proto_dist = []
    for k, ch in enumerate(chain_order):
        proto_dist.append(
            np.linalg.norm(center_of_mass[next_chain[k]] - center_of_mass[ch])
        )  # compute separation distances.
    separation_std = np.std(proto_dist)  # the standard deviation of the distance separations.
    return separation_std


def get_coord_CA(file_path):
    """
    Extracts Cα atom coordinates for each chain from a PDB or CIF file.
    Returns a numpy array with columns: [chain_id, residue_id, x, y, z]
    """
    import os
    from Bio.PDB import PDBParser, MMCIFParser

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdb":
        parser = PDBParser(QUIET=True)
    elif ext == ".cif":
        parser = MMCIFParser()
    else:
        raise ValueError("Unsupported file format: {}".format(ext))

    structure = parser.get_structure("structure", file_path)
    coords = []
    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if "CA" in residue:
                    atom = residue["CA"]
                    coords.append([chain_id, residue.id[1], atom.coord[0], atom.coord[1], atom.coord[2], ])
    return np.array(coords, dtype=object)


def _compute_chain_center_of_mass(coordinates):
    # coordinates: object array [chain_id, residue_id, x, y, z]
    chains = set(coordinates[:, 0])
    center_of_mass = {}
    for ch in chains:
        xyz = np.asarray(coordinates[coordinates[:, 0] == ch][:, 2:], dtype=float)
        center_of_mass[ch] = xyz.mean(axis=0)
    return center_of_mass


class Point:
    # 通过定义__init__方法， 在创建实例的时候把x,y,z的属性绑定，self表示创建的实例本身
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def getCircle(p1, p2, p3):
    A1 = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y)
    B1 = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z)
    C1 = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    D1 = -(A1 * p1.x + B1 * p1.y + C1 * p1.z)
    # A1x+B1y+C1z+D1=0为三个点所在的平面

    A2 = 2 * (p2.x - p1.x)
    B2 = 2 * (p2.y - p1.y)
    C2 = 2 * (p2.z - p1.z)
    D2 = (p1.x * p1.x + p1.y * p1.y + p1.z * p1.z - p2.x * p2.x - p2.y * p2.y - p2.z * p2.z)

    A3 = 2 * (p3.x - p1.x)
    B3 = 2 * (p3.y - p1.y)
    C3 = 2 * (p3.z - p1.z)
    D3 = (p1.x * p1.x + p1.y * p1.y + p1.z * p1.z - p3.x * p3.x - p3.y * p3.y - p3.z * p3.z)

    w = [[A1, B1, C1], [A2, B2, C2], [A3, B3, C3]]

    b = [-D1, -D2, -D3]
    center = np.linalg.solve(w, b)
    r = math.sqrt((p1.x - center[0]) * (p1.x - center[0]) + (p1.y - center[1]) * (p1.y - center[1]) + (p1.z - center[2]) * (p1.z - center[2]))
    return center, r


@loss_registry.register(required_params=["target_radius"], )
def circle_dual_cyclic(eval_result, target_radius, exclude_terminal_residues=5):
    result = {}
    for oligo in eval_result.names:
        result[oligo] = calculate_circle_dual_cyclic_single(oligo, eval_result, target_radius, exclude_terminal_residues)
    return result


def calculate_circle_dual_cyclic_single(oligo, eval_result, target_radius, exclude_terminal_residues=5):
    file_path = eval_result.complex_paths[oligo]
    coordinates = get_coord_CA(file_path)
    chains = set(coordinates[:, 0])

    # Compare distances between adjacent chains, including first-last.

    sampled_chains = random.sample(sorted(chains), 3)
    three_chain = np.empty((0, coordinates.shape[1]), dtype=object)
    for ch in sampled_chains:
        new_chainin = sorted(coordinates[coordinates[:, 0] == ch], key=lambda x: x[1])
        i = 0
        while i < len(new_chainin):
            new_chainin[i][1] = i + 1
            i += 1
        three_chain = np.vstack([three_chain, np.array(new_chainin, dtype=object)])
    resi = set(three_chain[:, 1])

    radi = {}
    for f in resi:
        p = three_chain[three_chain[:, 1] == f][:, 2:]
        p1 = p[0].tolist()
        p2 = p[1].tolist()
        p3 = p[-1].tolist()
        center, r = getCircle(Point(p1[0], p1[1], p1[2]), Point(p2[0], p2[1], p2[2]), Point(p3[0], p3[1], p3[2]), )
        radi[f] = r
    popKeys = (list(radi.keys())[:exclude_terminal_residues] + list(radi.keys())[-exclude_terminal_residues:])
    [radi.pop(k) for k in popKeys]
    rmid = np.median(list(radi.values()))
    rmin = min(list(radi.values()))
    if rmin < target_radius:
        rmin = target_radius
    return rmin


def _ordered_chain_centroids(coordinates):
    center_of_mass = _compute_chain_center_of_mass(coordinates)
    chain_order = sorted(center_of_mass.keys())
    if not chain_order:
        return [], np.empty((0, 3), dtype=float)
    centroids = np.stack([center_of_mass[ch] for ch in chain_order], axis=0)
    return chain_order, centroids


def _coefficient_of_variation(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) < 1e-8:
        return 0.0
    return float(np.std(arr) / mean)


def _angle_sorted_indices(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1] if vh.shape[0] > 1 else np.array([0.0, 1.0, 0.0])
    projected = np.stack([centered @ axis_u, centered @ axis_v], axis=1)
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    return np.argsort(angles)


def _shortest_cycle_indices(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    if n <= 2:
        return np.arange(n, dtype=int)
    fixed = 0
    best_order = None
    best_total = None
    for perm in itertools.permutations(range(1, n)):
        order = [fixed, *perm]
        if n > 2 and order[1] > order[-1]:
            continue
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += float(np.linalg.norm(points[order[j]] - points[order[i]]))
        if best_total is None or total < best_total:
            best_total = total
            best_order = order
    return np.asarray(best_order, dtype=int)


def _adjacent_edge_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.empty((0,), dtype=float)
    if points.shape[0] <= 8:
        order = _shortest_cycle_indices(points)
    else:
        order = _angle_sorted_indices(points)
    ordered = points[order]
    return np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)


def _plane_rms_normalized(points: np.ndarray, scale: float) -> float:
    if points.shape[0] < 3:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    distances = np.abs(centered @ normal)
    denom = max(float(scale), 1e-8)
    return float(np.sqrt(np.mean(distances**2)) / denom)


def calculate_cyclic_symmetry_components_single(oligo, eval_result):
    file_path = eval_result.complex_paths[oligo]
    coordinates = get_coord_CA(file_path)
    _, protomer_coms = _ordered_chain_centroids(coordinates)
    if protomer_coms.shape[0] == 0:
        return {
            "radius_cv": 0.0,
            "plane_rms_norm": 0.0,
            "edge_cv": 0.0,
            "composite": 0.0,
        }

    global_com = protomer_coms.mean(axis=0)
    radii = np.linalg.norm(protomer_coms - global_com[None, :], axis=-1)
    edge_lengths = _adjacent_edge_lengths(protomer_coms)
    mean_edge = float(np.mean(edge_lengths)) if edge_lengths.size else float(np.mean(radii))
    radius_cv = _coefficient_of_variation(radii)
    edge_cv = _coefficient_of_variation(edge_lengths)
    plane_rms_norm = _plane_rms_normalized(protomer_coms, mean_edge)
    composite = float(np.sqrt(radius_cv**2 + plane_rms_norm**2 + edge_cv**2))
    return {
        "radius_cv": radius_cv,
        "plane_rms_norm": plane_rms_norm,
        "edge_cv": edge_cv,
        "composite": composite,
    }


@loss_registry.register(description="Coefficient of variation of protomer-centroid radii about global COM.")
def cyclic_radius_cv(eval_result):
    return {
        oligo: calculate_cyclic_symmetry_components_single(oligo, eval_result)["radius_cv"]
        for oligo in eval_result.names
    }


@loss_registry.register(description="Normalized RMS distance of protomer centroids to their best-fit plane.")
def cyclic_plane_rms_norm(eval_result):
    return {
        oligo: calculate_cyclic_symmetry_components_single(oligo, eval_result)["plane_rms_norm"]
        for oligo in eval_result.names
    }


@loss_registry.register(description="Coefficient of variation of adjacent protomer-centroid edge lengths.")
def cyclic_edge_cv(eval_result):
    return {
        oligo: calculate_cyclic_symmetry_components_single(oligo, eval_result)["edge_cv"]
        for oligo in eval_result.names
    }


@loss_registry.register(description="Composite cyclic symmetry mismatch from radius, plane, and edge regularity.", )
def cyclic_symmetry_composite(eval_result):
    return {
        oligo: calculate_cyclic_symmetry_composite_single(oligo, eval_result)
        for oligo in eval_result.names
    }


def calculate_cyclic_symmetry_composite_single(oligo, eval_result):
    return float(calculate_cyclic_symmetry_components_single(oligo, eval_result)["composite"])


@loss_registry.register(description="Std of protomer COM radii about global COM.", )
def separation_std(eval_result):
    result = {}
    for oligo in eval_result.names:
        result[oligo] = calculate_separation_std_single(oligo, eval_result)
    return result


def calculate_separation_std_single(oligo, eval_result):
    file_path = eval_result.complex_paths[oligo]
    coordinates = get_coord_CA(file_path)
    _, protomer_coms = _ordered_chain_centroids(coordinates)
    if protomer_coms.shape[0] == 0:
        return 0.0

    global_com = protomer_coms.mean(axis=0)
    radii = np.linalg.norm(protomer_coms - global_com[None, :], axis=-1)
    return float(np.std(radii))

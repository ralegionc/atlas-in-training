"""Sphere geometry: coordinates <-> unit vectors, great-circle distance."""
import numpy as np

from config import EARTH_KM


def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def to_latlon(xyz):
    xyz = np.asarray(xyz, float)
    r = np.linalg.norm(xyz, axis=-1)
    r = np.where(r > 0, r, 1.0)
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2] / r, -1, 1)))
    lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lat, lon


def gc_km(xyz_a, xyz_b):
    """Great-circle distance between directions (need not be unit length)."""
    a = xyz_a / np.linalg.norm(xyz_a, axis=-1, keepdims=True)
    b = xyz_b / np.linalg.norm(xyz_b, axis=-1, keepdims=True)
    # atan2 form: accurate for tiny and near-antipodal angles alike
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, -1)
    return EARTH_KM * np.arctan2(cross, dot)

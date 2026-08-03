"""Generate the convex rounded-box collision mesh used by the FlipUp book.

The shape is the Minkowski sum of an inset box and a sphere.  Sampling support
points in each normal octant and taking their convex hull keeps the checked-in
OBJ deterministic, watertight, and suitable for MuJoCo's convex mesh contact.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

DEFAULT_SIZE = np.array([0.15, 0.10, 0.025], dtype=np.float64)
DEFAULT_RADIUS = 0.0005
DEFAULT_SEGMENTS = 8
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "flipup"
    / "assets"
    / "custom"
    / "book2_blend"
    / "book_collision_rounded.obj"
)


def rounded_box_mesh(
    size: np.ndarray,
    radius: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and outward-facing triangles for a rounded box."""
    size = np.asarray(size, dtype=np.float64)
    if size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError(f"size must contain three positive values, got {size}")
    if not 0.0 < radius < float(np.min(size)) / 2.0:
        raise ValueError("radius must be positive and smaller than every half-size")
    if segments < 2:
        raise ValueError("segments must be at least 2")

    center = size / 2.0
    inset = center - radius
    points: list[np.ndarray] = []
    for signs_tuple in itertools.product((-1.0, 1.0), repeat=3):
        signs = np.asarray(signs_tuple)
        for i in range(segments + 1):
            for j in range(segments + 1 - i):
                k = segments - i - j
                normal = np.asarray([i, j, k], dtype=np.float64)
                normal /= np.linalg.norm(normal)
                points.append(center + signs * (inset + radius * normal))

    vertices = np.unique(np.round(np.asarray(points), decimals=12), axis=0)
    hull = ConvexHull(vertices)
    faces = np.asarray(hull.simplices, dtype=np.int32).copy()
    for face in faces:
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if np.dot(normal, triangle.mean(axis=0) - center) < 0.0:
            face[1], face[2] = face[2], face[1]

    # Canonicalize face order to keep regeneration byte-for-byte stable.
    canonical_faces = []
    for face in faces:
        start = int(np.argmin(face))
        canonical_faces.append(np.roll(face, -start))
    faces = np.asarray(sorted(canonical_faces, key=tuple), dtype=np.int32)
    return vertices, faces


def write_obj(
    output: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    size: np.ndarray,
    radius: float,
    segments: int,
) -> None:
    lines = [
        "# Deterministic rounded-box collider for MuJoCo",
        f"# size={size.tolist()} radius={radius} segments={segments}",
        "o rounded_book_collision",
    ]
    lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    args = parser.parse_args()
    vertices, faces = rounded_box_mesh(DEFAULT_SIZE, args.radius, args.segments)
    write_obj(
        args.output,
        vertices,
        faces,
        size=DEFAULT_SIZE,
        radius=args.radius,
        segments=args.segments,
    )
    print(f"wrote {len(vertices)} vertices and {len(faces)} faces to {args.output}")


if __name__ == "__main__":
    main()

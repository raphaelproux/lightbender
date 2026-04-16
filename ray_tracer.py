"""2D ray–polyline intersection."""

import numpy as np


def _incidence_angle(d: np.ndarray, segment: np.ndarray) -> float:
    """Return the angle (in radians) between ray direction *d* and the surface normal.

    The normal is obtained by rotating *segment* by 90°.
    The returned value is in [0, π/2]: 0 means the ray is hitting head-on
    (perpendicular to the surface, i.e. along the normal), π/2 means the
    ray is grazing (parallel to the surface).
    """
    # Normal to the segment: rotate tangent by 90°
    normal = np.array([-segment[1], segment[0]])
    cos_angle = np.dot(d, normal) / (np.linalg.norm(d) * np.linalg.norm(normal))
    # Angle between ray and normal
    angle_to_normal = np.arccos(np.clip(abs(cos_angle), 0.0, 1.0))
    return float(angle_to_normal)


def ray_curve_intersection(
    x: float,
    y: float,
    vx: float,
    vy: float,
    curve: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Return the closest intersection point between a ray and a polyline curve.

    Parameters
    ----------
    x, y : float
        Origin of the ray.
    vx, vy : float
        Direction vector of the ray (does not need to be normalised).
    curve : np.ndarray, shape (N, 2)
        Ordered list of (xc, yc) points that define the polyline.

    Returns
    -------
    tuple (point, angle, normal) where

    - *point* is an ndarray of shape (2,) – the intersection position.
    - *angle* is the incidence angle in radians between the ray direction and
      the local surface normal (0 = head-on / perpendicular, π/2 = grazing /
      parallel).
    - *normal* is the unit surface normal at the intersection, oriented
      towards the incoming ray.

    Returns ``None`` if the ray does not intersect the curve.
    """
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] < 2:
        raise ValueError("curve must be an (N, 2) array with N >= 2")

    # Ray origin and direction
    o = np.array([x, y], dtype=float)
    d = np.array([vx, vy], dtype=float)

    best_t = np.inf
    best_point = None
    best_segment = None

    # Test every consecutive segment A→B of the polyline
    for i in range(len(curve) - 1):
        a = curve[i]
        b = curve[i + 1]
        ab = b - a  # segment direction

        # Solve  o + t·d = a + s·ab   ⟹   [d | -ab] · [t, s]ᵀ = a - o
        # Determinant of the 2×2 system
        denom = d[0] * (-ab[1]) - d[1] * (-ab[0])  # = -(vx·aby - vy·abx)
        if abs(denom) < 1e-12:
            continue  # ray is parallel to this segment

        diff = a - o
        t = (diff[0] * (-ab[1]) - diff[1] * (-ab[0])) / denom
        s = (d[0] * diff[1] - d[1] * diff[0]) / denom

        # t >= 0  → intersection is in the ray's forward direction
        # 0 <= s <= 1  → intersection lies on the segment
        if t >= 0.0 and 0.0 <= s <= 1.0 and t < best_t:
            best_t = t
            best_point = o + t * d
            best_segment = ab

    if best_point is None:
        return None

    # Compute the oriented unit normal (pointing toward the incoming ray)
    normal = np.array([-best_segment[1], best_segment[0]], dtype=float)
    normal /= np.linalg.norm(normal)
    if np.dot(d, normal) > 0:
        normal = -normal  # flip so that it faces the incoming ray

    return best_point, _incidence_angle(d, best_segment), normal


def refract_ray(
    x: float,
    y: float,
    vx: float,
    vy: float,
    curve: np.ndarray,
    n1: float,
    n2: float,
) -> tuple[float, float, float, float] | None:
    """Refract a ray through a surface defined by a polyline curve.

    Applies Snell's law at the first intersection of the ray with the curve:
    ``sin(θ₂) = (n₁ / n₂) · sin(θ₁)``.

    Parameters
    ----------
    x, y : float
        Origin of the incoming ray.
    vx, vy : float
        Direction of the incoming ray (does not need to be normalised).
    curve : np.ndarray, shape (N, 2)
        Polyline defining the refracting surface.
    n1 : float
        Refractive index of the medium the ray is travelling in.
    n2 : float
        Refractive index of the medium on the other side of the surface.

    Returns
    -------
    tuple (x_out, y_out, vx_out, vy_out) describing the refracted ray,
    or ``None`` if the ray does not hit the surface or total internal
    reflection occurs.
    """
    hit = ray_curve_intersection(x, y, vx, vy, curve)
    if hit is None:
        return None

    point, theta1, normal = hit

    # Snell's law: n1·sin(θ1) = n2·sin(θ2)
    sin_theta2 = (n1 / n2) * np.sin(theta1)
    if abs(sin_theta2) > 1.0:
        return None  # total internal reflection

    theta2 = np.arcsin(sin_theta2)

    # --- Build the refracted direction vector (Snell in vector form) ---
    # d_hat: unit incoming direction
    d = np.array([vx, vy], dtype=float)
    d_hat = d / np.linalg.norm(d)

    # cos of incidence angle (positive, since normal faces the ray)
    cos_i = -np.dot(d_hat, normal)

    eta = n1 / n2
    cos_r = np.cos(theta2)

    # Vector form: d_refracted = η·d̂ + (η·cos_i − cos_r)·n̂
    d_refracted = eta * d_hat + (eta * cos_i - cos_r) * normal

    return (
        float(point[0]),
        float(point[1]),
        float(d_refracted[0]),
        float(d_refracted[1]),
    )

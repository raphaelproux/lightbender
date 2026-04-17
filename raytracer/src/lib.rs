use numpy::ndarray::{Array1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute the incidence angle between a ray direction and a segment.
///
/// The surface normal is obtained by rotating the segment tangent by 90 degrees.
/// Returns a value in [0, pi/2].
fn incidence_angle(dx: f64, dy: f64, seg_x: f64, seg_y: f64) -> f64 {
    // Normal to the segment: rotate tangent by 90 degrees
    let nx = -seg_y;
    let ny = seg_x;

    let d_len = (dx * dx + dy * dy).sqrt();
    let n_len = (nx * nx + ny * ny).sqrt();

    if d_len < 1e-15 || n_len < 1e-15 {
        return 0.0;
    }

    let cos_angle = (dx * nx + dy * ny) / (d_len * n_len);
    // Angle between ray and normal, clamped to [0, pi/2]
    cos_angle.abs().clamp(0.0, 1.0).acos()
}

/// Internal result of a ray-curve intersection (no Python types).
struct IntersectionResult {
    point: [f64; 2],
    angle: f64,
    normal: [f64; 2],
}

/// Core ray-curve intersection logic, operating on a ndarray ArrayView.
fn ray_curve_intersection_impl(
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    curve: &ArrayView2<f64>,
) -> Option<IntersectionResult> {
    let n_points = curve.nrows();
    if n_points < 2 || curve.ncols() != 2 {
        return None;
    }

    let mut best_t = f64::INFINITY;
    let mut best_point: Option<[f64; 2]> = None;
    let mut best_seg: Option<[f64; 2]> = None;

    // Test every consecutive segment A -> B of the polyline
    for i in 0..n_points - 1 {
        let ax = curve[[i, 0]];
        let ay = curve[[i, 1]];
        let bx = curve[[i + 1, 0]];
        let by = curve[[i + 1, 1]];

        let abx = bx - ax;
        let aby = by - ay;

        // Solve: o + t*d = a + s*ab
        // [d | -ab] * [t, s]^T = a - o
        // denom = vx*(-aby) - vy*(-abx) = -(vx*aby - vy*abx)
        let denom = vx * (-aby) - vy * (-abx);
        if denom.abs() < 1e-12 {
            continue; // ray is parallel to this segment
        }

        let diff_x = ax - x;
        let diff_y = ay - y;

        let t = (diff_x * (-aby) - diff_y * (-abx)) / denom;
        let s = (vx * diff_y - vy * diff_x) / denom;

        // t >= 0 means forward direction; 0 <= s <= 1 means on segment
        if t >= 0.0 && s >= 0.0 && s <= 1.0 && t < best_t {
            best_t = t;
            best_point = Some([x + t * vx, y + t * vy]);
            best_seg = Some([abx, aby]);
        }
    }

    let point = best_point?;
    let seg = best_seg?;

    // Compute the oriented unit normal (pointing toward the incoming ray)
    let mut nx = -seg[1];
    let mut ny = seg[0];
    let n_len = (nx * nx + ny * ny).sqrt();
    nx /= n_len;
    ny /= n_len;

    // Flip so that it faces the incoming ray
    if vx * nx + vy * ny > 0.0 {
        nx = -nx;
        ny = -ny;
    }

    let angle = incidence_angle(vx, vy, seg[0], seg[1]);

    Some(IntersectionResult {
        point,
        angle,
        normal: [nx, ny],
    })
}

/// Return the closest intersection point between a ray and a polyline curve.
///
/// Parameters
/// ----------
/// x, y : float
///     Origin of the ray.
/// vx, vy : float
///     Direction vector of the ray (does not need to be normalised).
/// curve : numpy.ndarray, shape (N, 2)
///     Ordered list of (xc, yc) points that define the polyline.
///
/// Returns
/// -------
/// tuple (point, angle, normal) or None.
///
/// - point: ndarray of shape (2,) -- the intersection position.
/// - angle: incidence angle in radians (0 = head-on, pi/2 = grazing).
/// - normal: unit surface normal at the intersection, oriented towards the
///   incoming ray, as ndarray of shape (2,).
#[pyfunction]
fn ray_curve_intersection<'py>(
    py: Python<'py>,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    curve: PyReadonlyArray2<f64>,
) -> Option<(Bound<'py, PyArray1<f64>>, f64, Bound<'py, PyArray1<f64>>)> {
    let curve_view = curve.as_array();
    let result = ray_curve_intersection_impl(x, y, vx, vy, &curve_view)?;

    let point = Array1::from_vec(result.point.to_vec()).into_pyarray(py);
    let normal = Array1::from_vec(result.normal.to_vec()).into_pyarray(py);
    Some((point, result.angle, normal))
}

/// Refract a ray through a surface defined by a polyline curve.
///
/// Applies Snell's law at the first intersection of the ray with the curve.
///
/// Parameters
/// ----------
/// x, y : float
///     Origin of the incoming ray.
/// vx, vy : float
///     Direction of the incoming ray (does not need to be normalised).
/// curve : numpy.ndarray, shape (N, 2)
///     Polyline defining the refracting surface.
/// n1 : float
///     Refractive index of the medium the ray is travelling in.
/// n2 : float
///     Refractive index of the medium on the other side of the surface.
///
/// Returns
/// -------
/// tuple (x_out, y_out, vx_out, vy_out) or None if total internal
/// reflection occurs or the ray does not hit the surface.
#[pyfunction]
fn refract_ray(
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    curve: PyReadonlyArray2<f64>,
    n1: f64,
    n2: f64,
) -> Option<(f64, f64, f64, f64)> {
    let curve_view = curve.as_array();
    let hit = ray_curve_intersection_impl(x, y, vx, vy, &curve_view)?;

    let theta1 = hit.angle;

    // Snell's law: n1*sin(theta1) = n2*sin(theta2)
    let sin_theta2 = (n1 / n2) * theta1.sin();
    if sin_theta2.abs() > 1.0 {
        return None; // total internal reflection
    }

    let theta2 = sin_theta2.asin();

    // Build the refracted direction vector (Snell in vector form)
    let d_len = (vx * vx + vy * vy).sqrt();
    let dx_hat = vx / d_len;
    let dy_hat = vy / d_len;

    let nx = hit.normal[0];
    let ny = hit.normal[1];

    // cos of incidence angle (positive, since normal faces the ray)
    let cos_i = -(dx_hat * nx + dy_hat * ny);

    let eta = n1 / n2;
    let cos_r = theta2.cos();

    // Vector form: d_refracted = eta * d_hat + (eta * cos_i - cos_r) * normal
    let factor = eta * cos_i - cos_r;
    let rx = eta * dx_hat + factor * nx;
    let ry = eta * dy_hat + factor * ny;

    Some((hit.point[0], hit.point[1], rx, ry))
}

/// Batch version: trace multiple rays through the same curve intersection.
///
/// Parameters
/// ----------
/// rays : list of (x, y, vx, vy) tuples
/// curve : numpy.ndarray, shape (N, 2)
///
/// Returns a list of (point, angle, normal) or None for each ray.
#[pyfunction]
fn ray_curve_intersection_batch<'py>(
    py: Python<'py>,
    rays: Vec<(f64, f64, f64, f64)>,
    curve: PyReadonlyArray2<f64>,
) -> Vec<Option<(Bound<'py, PyArray1<f64>>, f64, Bound<'py, PyArray1<f64>>)>> {
    let curve_view = curve.as_array();

    rays.into_iter()
        .map(|(x, y, vx, vy)| {
            let result = ray_curve_intersection_impl(x, y, vx, vy, &curve_view)?;
            let point = Array1::from_vec(result.point.to_vec()).into_pyarray(py);
            let normal = Array1::from_vec(result.normal.to_vec()).into_pyarray(py);
            Some((point, result.angle, normal))
        })
        .collect()
}

/// Batch version: refract multiple rays through the same surface.
///
/// Parameters
/// ----------
/// rays : list of (x, y, vx, vy) tuples
/// curve : numpy.ndarray, shape (N, 2)
/// n1, n2 : float
///
/// Returns a list of (x_out, y_out, vx_out, vy_out) or None for each ray.
#[pyfunction]
fn refract_ray_batch(
    rays: Vec<(f64, f64, f64, f64)>,
    curve: PyReadonlyArray2<f64>,
    n1: f64,
    n2: f64,
) -> Vec<Option<(f64, f64, f64, f64)>> {
    let curve_view = curve.as_array();

    rays.into_iter()
        .map(|(x, y, vx, vy)| {
            let hit = ray_curve_intersection_impl(x, y, vx, vy, &curve_view)?;

            let theta1 = hit.angle;
            let sin_theta2 = (n1 / n2) * theta1.sin();
            if sin_theta2.abs() > 1.0 {
                return None;
            }

            let theta2 = sin_theta2.asin();
            let d_len = (vx * vx + vy * vy).sqrt();
            let dx_hat = vx / d_len;
            let dy_hat = vy / d_len;
            let nx = hit.normal[0];
            let ny = hit.normal[1];
            let cos_i = -(dx_hat * nx + dy_hat * ny);
            let eta = n1 / n2;
            let cos_r = theta2.cos();
            let factor = eta * cos_i - cos_r;
            let rx = eta * dx_hat + factor * nx;
            let ry = eta * dy_hat + factor * ny;

            Some((hit.point[0], hit.point[1], rx, ry))
        })
        .collect()
}

/// Native Rust ray tracer module for lightbender.
#[pymodule]
fn _raytracer_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ray_curve_intersection, m)?)?;
    m.add_function(wrap_pyfunction!(refract_ray, m)?)?;
    m.add_function(wrap_pyfunction!(ray_curve_intersection_batch, m)?)?;
    m.add_function(wrap_pyfunction!(refract_ray_batch, m)?)?;
    Ok(())
}

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    from ray_tracer import refract_ray, ray_curve_intersection
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    return mo, np, plt, ray_curve_intersection, refract_ray


@app.cell
def _(mo, np):
    interface_angle_slider = mo.ui.slider(label='Interface Angle', start=0, stop=np.pi/2, step=0.1, value=0, debounce=False)
    return (interface_angle_slider,)


@app.cell
def _(interface_angle_slider, np, ray_curve_intersection, refract_ray):
    start_ray = (-10.0, 0.0, 1.0, 0.0)  # Starting ray: (x, y, dx, dy)
    interface_angle = interface_angle_slider.value
    interface = np.array([[-2*np.sin(interface_angle), -2*np.cos(interface_angle)], [2*np.sin(interface_angle), 2*np.cos(interface_angle)]])

    new_ray = refract_ray(*start_ray, interface, n1=1.0, n2=1.5)

    end_interface = np.array([[5, -5], [5, 5]])
    end_point, _, _ = ray_curve_intersection(*new_ray, end_interface)
    return end_point, interface, new_ray, start_ray


@app.cell
def _(
    end_point,
    interface,
    interface_angle_slider,
    mo,
    new_ray,
    plt,
    start_ray,
):
    plt.plot([start_ray[0], new_ray[0], end_point[0]], [start_ray[1], new_ray[1], end_point[1]], label='Refracted Ray')
    plt.plot([interface[0, 0], interface[1, 0]], [interface[0, 1], interface[1, 1]], label='Interface')
    plt.gca().set_aspect('equal', adjustable='box')
    ax = mo.ui.matplotlib(plt.gca())
    mo.hstack([ax, interface_angle_slider], justify="start", align="center", gap=1)
    return


@app.cell
def _(np):
    def define_point_source(nb_rays=20, x=0.0, y=0.0):
        thetas = np.linspace(0, 2*np.pi, nb_rays, endpoint=False)
        return [(x, y, np.cos(theta), np.sin(theta)) for theta in thetas]

    def define_plane_wave_source(nb_rays=20, x_start=0.0, y_start=-10.0, x_end=0.0, y_end=10.0, angle=0.0):
        xs = np.linspace(x_start, x_end, nb_rays)
        ys = np.linspace(y_start, y_end, nb_rays)
        return [(x, y, np.cos(angle), np.sin(angle)) for x, y in zip(xs, ys)]

    def define_sphere(center_x=0.0, center_y=0.0, radius=1.0):
        thetas = np.linspace(0, 2*np.pi, 100)
        return np.array([(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)) for theta in thetas])

    def define_sphere_arc(apex_x=0.0, apex_y=0.0, radius=1.0, start_angle=0.0, end_angle=np.pi):
        thetas = np.linspace(start_angle, end_angle, 100)
        return np.array([(apex_x + radius * np.cos(theta) - radius, apex_y + radius * np.sin(theta)) for theta in thetas])

    return (
        define_plane_wave_source,
        define_point_source,
        define_sphere,
        define_sphere_arc,
    )


@app.cell
def _(np):
    def rotate_around_point(angle, start_x, start_y, rotate_around_x, rotate_around_y):
        shifted_x = start_x - rotate_around_x
        shifted_y = start_y - rotate_around_y
        rotated_x = shifted_x* np.cos(angle) - shifted_y * np.sin(angle)
        rotated_y = shifted_x* np.sin(angle) + shifted_y * np.cos(angle)
        return rotated_x + rotate_around_x, rotated_y + rotate_around_y

    return (rotate_around_point,)


@app.cell
def _(mo, np):
    nb_rays_slider = mo.ui.slider(label='Number of Rays', start=1, stop=1000, step=10, value=21, debounce=False)
    _steps = np.logspace(-1, 1, 200)-0.1
    x_offset_slider = mo.ui.slider(label='Travel right', steps=_steps, value=0, debounce=False)
    return nb_rays_slider, x_offset_slider


@app.cell
def _(
    define_point_source,
    define_sphere,
    nb_rays_slider,
    ray_curve_intersection,
):
    point_source_pos = (0.0, 0.0)
    point_source_ray_bundle = define_point_source(nb_rays=nb_rays_slider.value, x=point_source_pos[0], y=point_source_pos[1])

    intersections = [ray_curve_intersection(*ray, define_sphere(center_x=point_source_pos[0], center_y=point_source_pos[1], radius=5.0))[0] for ray in point_source_ray_bundle]
    return intersections, point_source_pos


@app.cell
def _(
    intersections,
    mo,
    nb_rays_slider,
    plt,
    point_source_pos,
    x_offset_slider,
):
    plt.figure(figsize=(15, 6))
    for intersection in intersections:
        plt.plot([point_source_pos[0], intersection[0]], [point_source_pos[1], intersection[1]], 'r')  # Mark intersection points
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_xlim(-0.05+x_offset_slider.value, 0.05+x_offset_slider.value)
    plt.gca().set_ylim(-0.02, 0.02)
    plt.tight_layout()

    ax2 = mo.ui.matplotlib(plt.gca())

    _sliders = mo.vstack([nb_rays_slider, x_offset_slider], gap=1)
    mo.hstack([ax2, _sliders], justify="start", align="center", gap=1)
    return


@app.cell
def _(mo, np):
    _diameters = list(np.linspace(0.5, 10.0, num=21))
    _radius_of_curvatures = list(np.linspace(1.0, 20.0, 21)) + [1e10]
    _thicknesses = list(np.linspace(0.2, 10.0, 10))
    diameter_slider = mo.ui.slider(label='Diameter', steps=_diameters, value=_diameters[0], debounce=False)
    thickness_slider = mo.ui.slider(label='Thickness', steps=_thicknesses, value=_thicknesses[0], debounce=False)
    radius_of_curvature_slider = mo.ui.slider(label='Radius of Curvature', steps=_radius_of_curvatures, value=_radius_of_curvatures[-1], debounce=False)
    angle_slider = mo.ui.slider(label='Angle', steps=list(np.linspace(-50, 50.0, 51)), value=0.0, debounce=False)
    nb_rays_plane_wave_slider = mo.ui.slider(label='Number of Rays', start=1, stop=130, step=2, value=71, debounce=False)
    return (
        angle_slider,
        diameter_slider,
        nb_rays_plane_wave_slider,
        radius_of_curvature_slider,
        thickness_slider,
    )


@app.cell
def _(
    angle_slider,
    define_plane_wave_source,
    define_sphere_arc,
    diameter_slider,
    nb_rays_plane_wave_slider,
    np,
    radius_of_curvature_slider,
    ray_curve_intersection,
    refract_ray,
    rotate_around_point,
    thickness_slider,
):
    diameter = diameter_slider.value
    radius_of_curvature = radius_of_curvature_slider.value

    ray_bundle_start_point = rotate_around_point(np.radians(angle_slider.value), start_x=-10.0, start_y=-10.0, rotate_around_x=0.0, rotate_around_y=0.0)
    ray_bundle_end_point = rotate_around_point(np.radians(angle_slider.value), start_x=-10.0, start_y=10.0, rotate_around_x=0.0, rotate_around_y=0.0)
    _start_ray_bundle = define_plane_wave_source(nb_rays=nb_rays_plane_wave_slider.value, x_start=ray_bundle_start_point[0], y_start=ray_bundle_start_point[1], x_end=ray_bundle_end_point[0], y_end=ray_bundle_end_point[1], angle=np.radians(angle_slider.value))

    plane = np.array([[0, -diameter/2], [0, diameter/2]])
    sphere_arc = define_sphere_arc(apex_x=thickness_slider.value, apex_y=0.0, radius=radius_of_curvature, start_angle=-np.arcsin(diameter/(2*radius_of_curvature)), end_angle=np.arcsin(diameter/(2*radius_of_curvature)))

    rays_positions = []
    for _start_ray in _start_ray_bundle:
        rays_positions.append([_start_ray[0:2]])
        _ray_1 = refract_ray(*_start_ray, plane, n1=1.0, n2=1.5)
        if _ray_1 is None:
            _intercept_point = ray_curve_intersection(*_start_ray, [[0, -1e10], [0, 1e10]])
            if _intercept_point is not None:
                rays_positions[-1].append(_intercept_point[0])
            continue
        else:
            rays_positions[-1].append(_ray_1[0:2])
        _ray_2 = refract_ray(*_ray_1, sphere_arc, n1=1.5, n2=1.0)
        if _ray_2 is None:
            _intercept_point = ray_curve_intersection(*_ray_1, sphere_arc)
            if _intercept_point is not None:
                rays_positions[-1].append(_intercept_point[0])
            continue
        else:
            rays_positions[-1].append(_ray_2[0:2])
        _final_pos = ray_curve_intersection(*_ray_2, [[30, -1e10], [30, 1e10]])[0]
        if _final_pos is None:
            continue
        else:
            rays_positions[-1].append(_final_pos[0:2])
    return plane, rays_positions, sphere_arc


@app.cell
def _(
    angle_slider,
    diameter_slider,
    mo,
    nb_rays_plane_wave_slider,
    np,
    plane,
    plt,
    radius_of_curvature_slider,
    rays_positions,
    sphere_arc,
    thickness_slider,
):
    plt.figure(figsize=(15,9))
    plt.plot(plane[:, 0], plane[:, 1], 'k')
    plt.plot([0, 0], [plane[-1, 1], 10], 'k', linewidth=3, solid_capstyle='butt')
    plt.plot([0, 0], [plane[0, 1], -10], 'k', linewidth=3, solid_capstyle='butt')
    plt.plot(sphere_arc[:, 0], sphere_arc[:, 1], 'k')
    plt.plot([plane[0, 0], sphere_arc[0, 0]], [plane[-1, 1], sphere_arc[-1, 1]], 'k')
    plt.plot([plane[-1, 0], sphere_arc[-1, 0]], [plane[0, 1], sphere_arc[0, 1]], 'k')
    for _ray in rays_positions:
        _ray = np.array(_ray)
        plt.plot(_ray[:, 0], _ray[:, 1], 'r', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_xlim(-10, 30)
    plt.gca().set_ylim(-10, 10)
    _sliders = mo.vstack([diameter_slider, radius_of_curvature_slider, thickness_slider, nb_rays_plane_wave_slider, angle_slider], gap=1)
    mo.hstack([plt.gca(), _sliders], justify="start", align="center", gap=1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

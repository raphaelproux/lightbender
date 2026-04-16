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

    def define_sphere(center_x=0.0, center_y=0.0, radius=1.0):
        thetas = np.linspace(0, 2*np.pi, 100)
        return np.array([(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)) for theta in thetas])

    return define_point_source, define_sphere


@app.cell
def _(mo):
    nb_rays_slider = mo.ui.slider(label='Number of Rays', start=1, stop=100, step=1, value=20, debounce=True)
    return (nb_rays_slider,)


@app.cell
def _(
    define_point_source,
    define_sphere,
    nb_rays_slider,
    ray_curve_intersection,
):
    point_source_pos = (0.0, 0.0)
    point_source_ray_bundle = define_point_source(nb_rays=nb_rays_slider.value, x=point_source_pos[0], y=point_source_pos[1])

    intersections = [ray_curve_intersection(*ray, define_sphere(center_x=point_source_pos[0], center_y=point_source_pos[1], radius=2.0))[0] for ray in point_source_ray_bundle]
    return intersections, point_source_pos


@app.cell
def _(intersections, mo, nb_rays_slider, plt, point_source_pos):
    for intersection in intersections:
    
        plt.plot([point_source_pos[0], intersection[0]], [point_source_pos[1], intersection[1]], 'r')  # Mark intersection points
    plt.gca().set_aspect('equal', adjustable='box')
    ax2 = mo.ui.matplotlib(plt.gca())
    mo.hstack([ax2, nb_rays_slider], justify="start", align="center", gap=1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

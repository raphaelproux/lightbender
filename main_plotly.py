import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Why do we need lenses?
    """)
    return


@app.cell
def _():
    from ray_tracer import refract_ray, ray_curve_intersection
    import numpy as np
    import marimo as mo
    import plotly.graph_objects as go

    from PIL import Image, ImageOps

    return Image, ImageOps, go, mo, np, ray_curve_intersection, refract_ray


@app.cell
def _(PLOTLY_TEMPLATE, go, np):

    def define_point_source(nb_rays=20, x=0.0, y=0.0, theta_start=0.0, theta_end=2 * np.pi):
        if theta_end - theta_start % 2 * np.pi == 0.0:
            endpoint = False
        else:
            endpoint = True
        thetas = np.linspace(theta_start, theta_end, nb_rays, endpoint=endpoint)
        return [(x, y, np.cos(theta), np.sin(theta)) for theta in thetas]


    def define_plane_wave_source(
        nb_rays=20, x_start=0.0, y_start=-10.0, x_end=0.0, y_end=10.0, angle=0.0
    ):
        xs = np.linspace(x_start, x_end, nb_rays)
        ys = np.linspace(y_start, y_end, nb_rays)
        return [(x, y, np.cos(angle), np.sin(angle)) for x, y in zip(xs, ys)]


    def define_sphere(center_x=0.0, center_y=0.0, radius=1.0):
        thetas = np.linspace(0, 2 * np.pi, 100)
        return np.array(
            [
                (center_x + radius * np.cos(theta), center_y + radius * np.sin(theta))
                for theta in thetas
            ]
        )


    def define_sphere_arc(
        apex_x=0.0, apex_y=0.0, radius=1.0, start_angle=0.0, end_angle=np.pi
    ):
        thetas = np.linspace(start_angle, end_angle, 100)
        return np.array(
            [
                (apex_x + radius * np.cos(theta) - radius, apex_y + radius * np.sin(theta))
                for theta in thetas
            ]
        )


    def rotate_around_point(angle, start_x, start_y, rotate_around_x, rotate_around_y):
        shifted_x = start_x - rotate_around_x
        shifted_y = start_y - rotate_around_y
        rotated_x = shifted_x * np.cos(angle) - shifted_y * np.sin(angle)
        rotated_y = shifted_x * np.sin(angle) + shifted_y * np.cos(angle)
        return rotated_x + rotate_around_x, rotated_y + rotate_around_y


    def lines_trace(segments, color="red", width=1, name=None):
        """Build a single Plotly Scatter trace for many disconnected line segments."""
        xs, ys = [], []
        for seg in segments:
            seg = np.asarray(seg)
            xs.extend(seg[:, 0].tolist() + [None])
            ys.extend(seg[:, 1].tolist() + [None])
        return go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            showlegend=name is not None,
        )


    def apply_template(fig):
        fig.update_layout(template=PLOTLY_TEMPLATE)
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)



    return (
        apply_template,
        define_plane_wave_source,
        define_point_source,
        define_sphere,
        define_sphere_arc,
        lines_trace,
        rotate_around_point,
    )


@app.cell
def _():
    PLOTLY_LAYOUT = dict(
        dragmode="pan",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    PLOTLY_CONFIG = dict(
        scrollZoom=True,
        displayModeBar=False,
    )
    PLOTLY_TEMPLATE = "plotly_white"
    return PLOTLY_LAYOUT, PLOTLY_TEMPLATE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Image from a point source
    """)
    return


@app.cell
def _(mo, np):
    _diameters = list(np.linspace(0.8, 15.0, num=21))
    _radius_of_curvatures = list(np.linspace(1.0, 20.0, 21)) + [1e10]
    _thicknesses = [0.01] + list(np.linspace(1.0, 10.0, 10))
    diameter_slider = mo.ui.slider(label='Diameter', steps=_diameters, value=_diameters[0], debounce=False)
    thickness_slider = mo.ui.slider(label='Thickness', steps=_thicknesses, value=_thicknesses[0], debounce=False)
    radius_of_curvature_slider = mo.ui.slider(label='Radius of Curvature', steps=_radius_of_curvatures, value=_radius_of_curvatures[-1], debounce=False)
    obj_pos_slider = mo.ui.slider(label='Object height', steps=list(np.linspace(-13, 13, 53)), value=0.0, debounce=False)
    nb_rays_point_source_slider = mo.ui.slider(label='Number of Rays', start=11, stop=250, step=2, value=101, debounce=False)
    return (
        diameter_slider,
        nb_rays_point_source_slider,
        obj_pos_slider,
        radius_of_curvature_slider,
        thickness_slider,
    )


@app.cell
def _(mo):
    _md = """ """
    mo.hstack([mo.image(src=r"assets/spotlight-beam-cutting-through-fog-dark-floor_1093951-27173.avif", width=500), mo.md(_md)], justify="start", align="center", gap=1)
    return


@app.cell
def _(
    Image,
    ImageOps,
    PLOTLY_LAYOUT,
    apply_template,
    define_point_source,
    define_sphere_arc,
    diameter_slider,
    go,
    lines_trace,
    mo,
    nb_rays_point_source_slider,
    np,
    obj_pos_slider,
    radius_of_curvature_slider,
    ray_curve_intersection,
    refract_ray,
    thickness_slider,
):
    _diameter = diameter_slider.value
    _radius_of_curvature = radius_of_curvature_slider.value

    _start_ray_bundle = define_point_source(nb_rays=nb_rays_point_source_slider.value, x=-30.0, y=obj_pos_slider.value, theta_start=-np.pi/6, theta_end=np.pi/6)

    _plane = np.array([[0, -_diameter/2], [0, _diameter/2]])
    _sphere_arc = define_sphere_arc(apex_x=thickness_slider.value, apex_y=0.0, radius=_radius_of_curvature, start_angle=-np.arcsin(_diameter/(2*_radius_of_curvature)), end_angle=np.arcsin(_diameter/(2*_radius_of_curvature)))

    _rays_positions = []
    for _start_ray in _start_ray_bundle:
        _rays_positions.append([_start_ray[0:2]])
        _ray_1 = refract_ray(*_start_ray, curve=_plane, n1=1.0, n2=1.5)
        if _ray_1 is None:
            _intercept_point = ray_curve_intersection(*_start_ray, [[0, -1e10], [0, 1e10]])
            if _intercept_point is not None:
                _rays_positions[-1].append(_intercept_point[0])
            continue
        else:
            _rays_positions[-1].append(_ray_1[0:2])
        _ray_2 = refract_ray(*_ray_1, curve=_sphere_arc, n1=1.5, n2=1.0)
        if _ray_2 is None:
            _intercept_point = ray_curve_intersection(*_ray_1, _sphere_arc)
            if _intercept_point is not None:
                _rays_positions[-1].append(_intercept_point[0])
            continue
        else:
            _rays_positions[-1].append(_ray_2[0:2])
        _final_pos = ray_curve_intersection(*_ray_2, [[30, -1e10], [30, 1e10]])[0]
        if _final_pos is None:
            continue
        else:
            _rays_positions[-1].append(_final_pos[0:2])

    _lens_parts = [
        _plane,
        _sphere_arc,
        np.array([[_plane[0, 0], _plane[-1, 1]], [_sphere_arc[0, 0], _sphere_arc[-1, 1]]]),
        np.array([[_plane[-1, 0], _plane[0, 1]], [_sphere_arc[-1, 0], _sphere_arc[0, 1]]]),
    ]
    _screen_parts = [
        np.array([[0, _plane[-1, 1]], [0, 30]]),
        np.array([[0, _plane[0, 1]], [0, -30]]),
    ]

    _palm_tree = Image.open("assets\Image1.png")

    _fig = go.Figure(
        layout={
            **PLOTLY_LAYOUT,
            "xaxis": {
                **PLOTLY_LAYOUT["xaxis"],
                "range": [-35, 20],
                "autorange": False,
                "constrain": "domain",
            },
            "yaxis": {
                **PLOTLY_LAYOUT["yaxis"],
                "range": [-20, 20],
                "autorange": False,
                "constrain": "domain",
            },
            "height": 600,
            "width": 1100,"showlegend":False,
        }
    )
    _fig.add_trace(lines_trace(_lens_parts, color="black", width=1.5, name="Lens"))
    _fig.add_trace(lines_trace(_rays_positions, color="limegreen", width=0.8, name="Rays"))
    _fig.add_trace(lines_trace(_screen_parts, color="black", width=5, name="Screen"))
    _fig.add_trace(lines_trace([np.array([[15, -10], [15, 10]])], color="black", width=3, name="Detector"))
    _fig.add_layout_image(dict(source=_palm_tree,
            xref="x",
            yref="y",
            x=-29,
            y=0,
            xanchor="center",
            yanchor="middle",
            sizex=20,
            sizey=25,layer="below"))
    _fig.add_layout_image(dict(source=ImageOps.flip(_palm_tree),
            xref="x",
            yref="y",
            x=16,
            y=0,
            xanchor="center",
            yanchor="middle",
            sizex=12,
            sizey=12,layer="below", opacity=0.5))
    apply_template(_fig)

    _sliders = mo.vstack([diameter_slider, radius_of_curvature_slider, thickness_slider, nb_rays_point_source_slider, obj_pos_slider], gap=1)
    mo.hstack([_fig, _sliders], justify="start", align="center", gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Refraction
    """)
    return


@app.cell
def _(mo):
    mo.image(src=r"assets/Refraction_photo.png", width=500)
    return


@app.cell
def _(mo):
    interface_angle_slider = mo.ui.slider(label='Interface Angle (deg)', start=0, stop=90, step=1, value=0, debounce=False)
    n1_slider = mo.ui.slider(label='Refractive Index n1', start=1.0, stop=2.0, step=0.1, value=1.0, debounce=False)
    n2_slider = mo.ui.slider(label='Refractive Index n2', start=1.0, stop=2.0, step=0.1, value=1.5, debounce=False)
    return interface_angle_slider, n1_slider, n2_slider


@app.cell
def _(
    PLOTLY_LAYOUT,
    apply_template,
    go,
    interface_angle_slider,
    mo,
    n1_slider,
    n2_slider,
    np,
    ray_curve_intersection,
    refract_ray,
):
    _start_ray = (-10.0, 0.0, 1.0, 0.0)  # Starting ray: (x, y, dx, dy)
    _interface_angle = np.radians(interface_angle_slider.value)
    _interface = np.array([[-20*np.sin(_interface_angle), -20*np.cos(_interface_angle)], [20*np.sin(_interface_angle), 20*np.cos(_interface_angle)]])

    _new_ray = refract_ray(*_start_ray, _interface, n1=n1_slider.value, n2=n2_slider.value)

    _end_interface = np.array([[10, -1e10], [10, 1e10]])
    _end_point, _, _ = ray_curve_intersection(*_new_ray, _end_interface)

    _fig = go.Figure(
        layout={
            **PLOTLY_LAYOUT,
            "xaxis": {
                **PLOTLY_LAYOUT["xaxis"],
                "range": [-10, 10],
                "autorange": False,
                "constrain": "domain",
            },
            "yaxis": {
                **PLOTLY_LAYOUT["yaxis"],
                "range": [-5, 5],
                "autorange": False,
                "constrain": "domain",
            },
            "height": 600,
        }
    )
    if _new_ray is not None:
        _end_interface = np.array([[10, -1e10], [10, 1e10]])
        _end_point, _, _ = ray_curve_intersection(*_new_ray, _end_interface)
        _fig.add_trace(
            go.Scattergl(
                x=[_start_ray[0], _new_ray[0], _end_point[0]],
                y=[_start_ray[1], _new_ray[1], _end_point[1]],
                mode="lines",
                line=dict(color="blue"),
                name="Refracted Ray",
            )
        )
    _fig.add_trace(
        go.Scattergl(
            x=[_interface[0, 0], _interface[1, 0], 20, 200],
            y=[_interface[0, 1], _interface[1, 1], 20, -200],
            mode="none",
            fill="toself",
            fillcolor="rgba(161, 203, 209, 0.5)",
            line=dict(color="orange"),
            name="Interface",
        )
    )
    apply_template(_fig)
    _sliders = mo.vstack([interface_angle_slider, n1_slider, n2_slider], gap=1)
    mo.hstack([_fig, _sliders], justify="start", align="center", gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Point source
    """)
    return


@app.cell
def _(mo, np):
    nb_rays_slider = mo.ui.slider(label='Number of Rays', start=1, stop=4000, step=10, value=21, debounce=False)
    _steps = np.logspace(-1, 0.7, 200)-0.1
    x_offset_slider = mo.ui.slider(label='Travel right', steps=_steps, value=0, debounce=False)
    return nb_rays_slider, x_offset_slider


@app.cell
def _(
    PLOTLY_LAYOUT,
    apply_template,
    define_point_source,
    define_sphere,
    go,
    lines_trace,
    mo,
    nb_rays_slider,
    np,
    ray_curve_intersection,
    x_offset_slider,
):
    _point_source_pos = (0.0, 0.0)
    _point_source_ray_bundle = define_point_source(nb_rays=nb_rays_slider.value, x=_point_source_pos[0], y=_point_source_pos[1])

    _intersections = [ray_curve_intersection(*_ray, define_sphere(center_x=_point_source_pos[0], center_y=_point_source_pos[1], radius=5.0))[0] for _ray in _point_source_ray_bundle]

    _ray_segs = [
        np.array([[_point_source_pos[0], _point_source_pos[1]], [_ix[0], _ix[1]]])
        for _ix in _intersections
    ]
    _fig = go.Figure(
        data=[lines_trace(_ray_segs, color="red", width=1)],
        layout={
            **PLOTLY_LAYOUT,
            "xaxis": {
                **PLOTLY_LAYOUT["xaxis"],
                "range": [-0.05 + x_offset_slider.value, 0.05 + x_offset_slider.value],
            },
            "yaxis": {**PLOTLY_LAYOUT["yaxis"], "range": [-0.02, 0.02]},
            "height": 400,
        },
    )
    apply_template(_fig)

    _sliders = mo.vstack([nb_rays_slider, x_offset_slider], gap=1)
    mo.hstack([_fig, _sliders], justify="start", align="center", gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Focus from infinity
    """)
    return


@app.cell
def _(mo):
    mo.image(src=r"assets/optical_layout.png", width=800, style={"margin": "auto"})
    return


@app.cell
def _(mo, np):
    _diameters = list(np.linspace(0.5, 10.0, num=21))
    _radius_of_curvatures = list(np.linspace(1.0, 20.0, 21)) + [1e10]
    _thicknesses = list(np.linspace(0.2, 10.0, 10))
    infinity_diameter_slider = mo.ui.slider(label='Diameter', steps=_diameters, value=_diameters[0], debounce=False)
    infinity_thickness_slider = mo.ui.slider(label='Thickness', steps=_thicknesses, value=_thicknesses[0], debounce=False)
    infinity_radius_of_curvature_slider = mo.ui.slider(label='Radius of Curvature', steps=_radius_of_curvatures, value=_radius_of_curvatures[-1], debounce=False)
    infinity_angle_slider = mo.ui.slider(label='Angle', steps=list(np.linspace(-50, 50.0, 51)), value=0.0, debounce=False)
    infinity_nb_rays_plane_wave_slider = mo.ui.slider(label='Number of Rays', start=1, stop=130, step=2, value=71, debounce=False)
    return (
        infinity_angle_slider,
        infinity_diameter_slider,
        infinity_nb_rays_plane_wave_slider,
        infinity_radius_of_curvature_slider,
        infinity_thickness_slider,
    )


@app.cell
def _(
    PLOTLY_LAYOUT,
    apply_template,
    define_plane_wave_source,
    define_sphere_arc,
    go,
    infinity_angle_slider,
    infinity_diameter_slider,
    infinity_nb_rays_plane_wave_slider,
    infinity_radius_of_curvature_slider,
    infinity_thickness_slider,
    lines_trace,
    mo,
    np,
    ray_curve_intersection,
    refract_ray,
    rotate_around_point,
):
    _diameter = infinity_diameter_slider.value
    _radius_of_curvature = infinity_radius_of_curvature_slider.value

    _ray_bundle_start_point = rotate_around_point(np.radians(infinity_angle_slider.value), start_x=-15.0, start_y=-10.0, rotate_around_x=0.0, rotate_around_y=0.0)
    _ray_bundle_end_point = rotate_around_point(np.radians(infinity_angle_slider.value), start_x=-15.0, start_y=10.0, rotate_around_x=0.0, rotate_around_y=0.0)
    _start_ray_bundle = define_plane_wave_source(nb_rays=infinity_nb_rays_plane_wave_slider.value, x_start=_ray_bundle_start_point[0], y_start=_ray_bundle_start_point[1], x_end=_ray_bundle_end_point[0], y_end=_ray_bundle_end_point[1], angle=np.radians(infinity_angle_slider.value))

    _plane = np.array([[0, -_diameter/2], [0, _diameter/2]])
    _sphere_arc = define_sphere_arc(apex_x=infinity_thickness_slider.value, apex_y=0.0, radius=_radius_of_curvature, start_angle=-np.arcsin(_diameter/(2*_radius_of_curvature)), end_angle=np.arcsin(_diameter/(2*_radius_of_curvature)))

    _rays_positions = []
    for _start_ray in _start_ray_bundle:
        _rays_positions.append([_start_ray[0:2]])
        _ray_1 = refract_ray(*_start_ray, curve=_plane, n1=1.0, n2=1.5)
        if _ray_1 is None:
            _intercept_point = ray_curve_intersection(*_start_ray, [[0, -1e10], [0, 1e10]])
            if _intercept_point is not None:
                _rays_positions[-1].append(_intercept_point[0])
            continue
        else:
            _rays_positions[-1].append(_ray_1[0:2])
        _ray_2 = refract_ray(*_ray_1, curve=_sphere_arc, n1=1.5, n2=1.0)
        if _ray_2 is None:
            _intercept_point = ray_curve_intersection(*_ray_1, _sphere_arc)
            if _intercept_point is not None:
                _rays_positions[-1].append(_intercept_point[0])
            continue
        else:
            _rays_positions[-1].append(_ray_2[0:2])
        _final_pos = ray_curve_intersection(*_ray_2, [[30, -1e10], [30, 1e10]])[0]
        if _final_pos is None:
            continue
        else:
            _rays_positions[-1].append(_final_pos[0:2])

    _lens_parts = [
        _plane,
        _sphere_arc,
        np.array([[_plane[0, 0], _plane[-1, 1]], [_sphere_arc[0, 0], _sphere_arc[-1, 1]]]),
        np.array([[_plane[-1, 0], _plane[0, 1]], [_sphere_arc[-1, 0], _sphere_arc[0, 1]]]),
    ]
    _screen_parts = [
        np.array([[0, _plane[-1, 1]], [0, 10]]),
        np.array([[0, _plane[0, 1]], [0, -10]]),
    ]

    _fig = go.Figure(
        layout={
            **PLOTLY_LAYOUT,
            "xaxis": {
                **PLOTLY_LAYOUT["xaxis"],
                "range": [-10, 30],
                "autorange": False,
                "constrain": "domain",
            },
            "yaxis": {
                **PLOTLY_LAYOUT["yaxis"],
                "range": [-10, 10],
                "autorange": False,
                "constrain": "domain",
            },
            "width":1000
        }
    )
    _fig.add_trace(lines_trace(_lens_parts, color="black", width=1, name="Lens"))
    _fig.add_trace(lines_trace(_screen_parts, color="black", width=6, name="Screen"))
    _fig.add_trace(lines_trace(_rays_positions, color="red", width=0.5, name="Rays"))
    apply_template(_fig)

    _sliders = mo.vstack([infinity_diameter_slider, infinity_radius_of_curvature_slider, infinity_thickness_slider, infinity_nb_rays_plane_wave_slider, infinity_angle_slider], gap=1)
    mo.hstack([_fig, _sliders], justify="center", align="center", gap=1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

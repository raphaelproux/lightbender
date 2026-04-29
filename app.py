import numpy as np
import panel as pn
import plotly.graph_objects as go

from ray_tracer import ray_curve_intersection, refract_ray

pn.extension("plotly", sizing_mode="stretch_width")

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


# --- Helper functions ---


def define_point_source(nb_rays=20, x=0.0, y=0.0):
    thetas = np.linspace(0, 2 * np.pi, nb_rays, endpoint=False)
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


# ==========================================================================
# Section 1: Simple Refraction
# ==========================================================================

interface_angle_slider = pn.widgets.FloatSlider(
    name="Interface Angle (deg)",
    start=0.0,
    end=90,
    step=0.1,
    value=0.0,
)
n1_slider = pn.widgets.FloatSlider(
    name="Refractive Index n1",
    start=1.0,
    end=2.0,
    step=0.1,
    value=1.0,
)
n2_slider = pn.widgets.FloatSlider(
    name="Refractive Index n2",
    start=1.0,
    end=2.0,
    step=0.1,
    value=1.5,
)


@pn.depends(interface_angle_slider, n1_slider, n2_slider)
def plot_simple_refraction(interface_angle, n1, n2):
    start_ray = (-10.0, 0.0, 1.0, 0.0)
    interface = np.array(
        [
            [
                -20 * np.sin(np.radians(interface_angle)),
                -20 * np.cos(np.radians(interface_angle)),
            ],
            [
                20 * np.sin(np.radians(interface_angle)),
                20 * np.cos(np.radians(interface_angle)),
            ],
        ]
    )
    new_ray = refract_ray(*start_ray, interface, n1=n1, n2=n2)

    fig = go.Figure(
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
    if new_ray is not None:
        end_interface = np.array([[10, -1e10], [10, 1e10]])
        end_point, _, _ = ray_curve_intersection(*new_ray, end_interface)
        fig.add_trace(
            go.Scattergl(
                x=[start_ray[0], new_ray[0], end_point[0]],
                y=[start_ray[1], new_ray[1], end_point[1]],
                mode="lines",
                line=dict(color="blue"),
                name="Refracted Ray",
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=[interface[0, 0], interface[1, 0], 20, 200],
            y=[interface[0, 1], interface[1, 1], 20, -200],
            mode="none",
            fill="toself",
            fillcolor="rgba(161, 203, 209, 0.5)",
            line=dict(color="orange"),
            name="Interface",
        )
    )
    apply_template(fig)
    return pn.pane.Plotly(fig, config=PLOTLY_CONFIG, sizing_mode="stretch_width")


section1 = pn.Column(
    "# Simple Refraction",
    pn.Row(
        plot_simple_refraction,
        pn.Column(interface_angle_slider, n1_slider, n2_slider, width=250),
    ),
)


# ==========================================================================
# Section 2: Point Source
# ==========================================================================

nb_rays_slider = pn.widgets.IntSlider(
    name="Number of Rays",
    start=1,
    end=4000,
    step=10,
    value=21,
)
_steps_list = (np.logspace(-1, 0.7, 200) - 0.1).tolist()
x_offset_slider = pn.widgets.DiscreteSlider(
    name="Travel right",
    options={f"{v:.2f}": v for v in _steps_list},
    value=0.0,
)


@pn.depends(nb_rays_slider, x_offset_slider)
def plot_point_source(nb_rays, x_offset):
    point_source_pos = (0.0, 0.0)
    ray_bundle = define_point_source(
        nb_rays=nb_rays, x=point_source_pos[0], y=point_source_pos[1]
    )
    intersections = [
        ray_curve_intersection(
            *ray,
            define_sphere(
                center_x=point_source_pos[0], center_y=point_source_pos[1], radius=5.0
            ),
        )[0]
        for ray in ray_bundle
    ]

    ray_segs = [
        np.array([[point_source_pos[0], point_source_pos[1]], [ix[0], ix[1]]])
        for ix in intersections
    ]
    fig = go.Figure(
        data=[lines_trace(ray_segs, color="red", width=1)],
        layout={
            **PLOTLY_LAYOUT,
            "xaxis": {
                **PLOTLY_LAYOUT["xaxis"],
                "range": [-0.05 + x_offset, 0.05 + x_offset],
            },
            "yaxis": {**PLOTLY_LAYOUT["yaxis"], "range": [-0.02, 0.02]},
            "height": 400,
        },
    )
    apply_template(fig)
    return pn.pane.Plotly(fig, config=PLOTLY_CONFIG, sizing_mode="stretch_width")


section2 = pn.Column(
    "# Point Source at infinity",
    pn.pane.Image("assets/optical_layout.png", width=800),
    pn.Row(
        plot_point_source,
        pn.Column(nb_rays_slider, x_offset_slider, width=250),
    ),
)


# ==========================================================================
# Section 3: Plano-Convex Lens
# ==========================================================================

_diameters = list(np.linspace(0.5, 10.0, num=21))
_radius_of_curvatures = list(np.linspace(1.0, 20.0, 21)) + [1e10]
_thicknesses = list(np.linspace(0.2, 10.0, 10))

diameter_slider = pn.widgets.DiscreteSlider(
    name="Diameter", options=_diameters, value=_diameters[0]
)
roc_slider = pn.widgets.DiscreteSlider(
    name="Radius of Curvature",
    options=_radius_of_curvatures,
    value=_radius_of_curvatures[-1],
)
thickness_slider = pn.widgets.DiscreteSlider(
    name="Thickness", options=_thicknesses, value=_thicknesses[0]
)
nb_rays_pw_slider = pn.widgets.IntSlider(
    name="Number of Rays", start=1, end=130, step=2, value=71
)
angle_slider = pn.widgets.DiscreteSlider(
    name="Angle", options=list(np.linspace(-50, 50.0, 51)), value=0.0
)


@pn.depends(
    diameter_slider, roc_slider, thickness_slider, nb_rays_pw_slider, angle_slider
)
def plot_lens(diameter, radius_of_curvature, thickness, nb_rays_plane_wave, angle):
    ray_bundle_start_point = rotate_around_point(
        np.radians(angle), -15.0, -10.0, 0.0, 0.0
    )
    ray_bundle_end_point = rotate_around_point(np.radians(angle), -15.0, 10.0, 0.0, 0.0)
    start_ray_bundle = define_plane_wave_source(
        nb_rays=nb_rays_plane_wave,
        x_start=ray_bundle_start_point[0],
        y_start=ray_bundle_start_point[1],
        x_end=ray_bundle_end_point[0],
        y_end=ray_bundle_end_point[1],
        angle=np.radians(angle),
    )

    plane = np.array([[0, -diameter / 2], [0, diameter / 2]])
    sphere_arc = define_sphere_arc(
        apex_x=thickness,
        apex_y=0.0,
        radius=radius_of_curvature,
        start_angle=-np.arcsin(diameter / (2 * radius_of_curvature)),
        end_angle=np.arcsin(diameter / (2 * radius_of_curvature)),
    )

    rays_positions = []
    for sr in start_ray_bundle:
        rays_positions.append([sr[0:2]])
        ray1 = refract_ray(*sr, plane, n1=1.0, n2=1.5)
        if ray1 is None:
            intercept = ray_curve_intersection(*sr, [[0, -1e10], [0, 1e10]])
            if intercept is not None:
                rays_positions[-1].append(intercept[0])
            continue
        else:
            rays_positions[-1].append(ray1[0:2])
        ray2 = refract_ray(*ray1, sphere_arc, n1=1.5, n2=1.0)
        if ray2 is None:
            intercept = ray_curve_intersection(*ray1, sphere_arc)
            if intercept is not None:
                rays_positions[-1].append(intercept[0])
            continue
        else:
            rays_positions[-1].append(ray2[0:2])
        final_pos = ray_curve_intersection(*ray2, [[30, -1e10], [30, 1e10]])[0]
        if final_pos is None:
            continue
        else:
            rays_positions[-1].append(final_pos[0:2])

    lens_parts = [
        plane,
        sphere_arc,
        np.array([[plane[0, 0], plane[-1, 1]], [sphere_arc[0, 0], sphere_arc[-1, 1]]]),
        np.array([[plane[-1, 0], plane[0, 1]], [sphere_arc[-1, 0], sphere_arc[0, 1]]]),
    ]
    screen_parts = [
        np.array([[0, plane[-1, 1]], [0, 10]]),
        np.array([[0, plane[0, 1]], [0, -10]]),
    ]

    fig = go.Figure(
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
            "height": 600,
        }
    )
    fig.add_trace(lines_trace(lens_parts, color="black", width=1, name="Lens"))
    fig.add_trace(lines_trace(screen_parts, color="black", width=6, name="Screen"))
    fig.add_trace(lines_trace(rays_positions, color="red", width=0.5, name="Rays"))
    apply_template(fig)
    return pn.pane.Plotly(fig, config=PLOTLY_CONFIG, sizing_mode="stretch_width")


section3 = pn.Column(
    "# Making images",
    pn.Row(
        plot_lens,
        pn.Column(
            diameter_slider,
            roc_slider,
            thickness_slider,
            nb_rays_pw_slider,
            angle_slider,
            width=250,
        ),
    ),
)

# ==========================================================================
# Section 4: Plano-Convex Lens with point source
# ==========================================================================

_diameters_point_source = list(np.linspace(0.5, 10.0, num=21))
_radius_of_curvatures_point_source = list(np.linspace(1.0, 20.0, 21)) + [1e10]
_thicknesses_point_source = list(np.linspace(0.2, 10.0, 10))

diameter_point_source_slider = pn.widgets.DiscreteSlider(
    name="Diameter", options=_diameters_point_source, value=_diameters_point_source[0]
)
roc_point_source_slider = pn.widgets.DiscreteSlider(
    name="Radius of Curvature",
    options=_radius_of_curvatures_point_source,
    value=_radius_of_curvatures_point_source[-1],
)
thickness_point_source_slider = pn.widgets.DiscreteSlider(
    name="Thickness",
    options=_thicknesses_point_source,
    value=_thicknesses_point_source[0],
)
nb_rays_pw_point_source_slider = pn.widgets.IntSlider(
    name="Number of Rays", start=1, end=130, step=2, value=71
)
angle_point_source_slider = pn.widgets.DiscreteSlider(
    name="Angle", options=list(np.linspace(-50, 50.0, 51)), value=0.0
)


@pn.depends(
    diameter_point_source_slider,
    roc_point_source_slider,
    thickness_point_source_slider,
    nb_rays_pw_point_source_slider,
    angle_point_source_slider,
)
def plot_lens_point_source(
    diameter, radius_of_curvature, thickness, nb_rays_plane_wave, angle
):
    ray_bundle_start_point = rotate_around_point(
        np.radians(angle), -15.0, -10.0, 0.0, 0.0
    )
    ray_bundle_end_point = rotate_around_point(np.radians(angle), -15.0, 10.0, 0.0, 0.0)
    start_ray_bundle = define_plane_wave_source(
        nb_rays=nb_rays_plane_wave,
        x_start=ray_bundle_start_point[0],
        y_start=ray_bundle_start_point[1],
        x_end=ray_bundle_end_point[0],
        y_end=ray_bundle_end_point[1],
        angle=np.radians(angle),
    )

    plane = np.array([[0, -diameter / 2], [0, diameter / 2]])
    sphere_arc = define_sphere_arc(
        apex_x=thickness,
        apex_y=0.0,
        radius=radius_of_curvature,
        start_angle=-np.arcsin(diameter / (2 * radius_of_curvature)),
        end_angle=np.arcsin(diameter / (2 * radius_of_curvature)),
    )

    rays_positions = []
    for sr in start_ray_bundle:
        rays_positions.append([sr[0:2]])
        ray1 = refract_ray(*sr, plane, n1=1.0, n2=1.5)
        if ray1 is None:
            intercept = ray_curve_intersection(*sr, [[0, -1e10], [0, 1e10]])
            if intercept is not None:
                rays_positions[-1].append(intercept[0])
            continue
        else:
            rays_positions[-1].append(ray1[0:2])
        ray2 = refract_ray(*ray1, sphere_arc, n1=1.5, n2=1.0)
        if ray2 is None:
            intercept = ray_curve_intersection(*ray1, sphere_arc)
            if intercept is not None:
                rays_positions[-1].append(intercept[0])
            continue
        else:
            rays_positions[-1].append(ray2[0:2])
        final_pos = ray_curve_intersection(*ray2, [[30, -1e10], [30, 1e10]])[0]
        if final_pos is None:
            continue
        else:
            rays_positions[-1].append(final_pos[0:2])

    lens_parts = [
        plane,
        sphere_arc,
        np.array([[plane[0, 0], plane[-1, 1]], [sphere_arc[0, 0], sphere_arc[-1, 1]]]),
        np.array([[plane[-1, 0], plane[0, 1]], [sphere_arc[-1, 0], sphere_arc[0, 1]]]),
    ]
    screen_parts = [
        np.array([[0, plane[-1, 1]], [0, 10]]),
        np.array([[0, plane[0, 1]], [0, -10]]),
    ]

    fig = go.Figure(
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
            "height": 600,
        }
    )
    fig.add_trace(lines_trace(lens_parts, color="black", width=1, name="Lens"))
    fig.add_trace(lines_trace(screen_parts, color="black", width=6, name="Screen"))
    fig.add_trace(lines_trace(rays_positions, color="red", width=0.5, name="Rays"))
    apply_template(fig)
    return pn.pane.Plotly(fig, config=PLOTLY_CONFIG, sizing_mode="stretch_width")


section4 = pn.Column(
    "# Making images of a point source",
    pn.Row(
        plot_lens_point_source,
        pn.Column(
            diameter_point_source_slider,
            roc_point_source_slider,
            thickness_point_source_slider,
            nb_rays_pw_point_source_slider,
            angle_point_source_slider,
            width=250,
        ),
    ),
)


# ==========================================================================
# Serve
# ==========================================================================

pn.Column(section1, section2, section3, section4).servable(title="Lightbender")

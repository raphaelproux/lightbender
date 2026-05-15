import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.interpolate import RegularGridInterpolator
    from zernike import RZern


    return RegularGridInterpolator, go, mo, np


@app.cell
def _(RegularGridInterpolator, np):
    def slice_3d_surface(x, y, z, x0, y0, angle_deg, num_points=100, t_range=10.0):
        """
        Take a slice through a 3D surface defined by mesh grids (x, y, z) at an arbitrary point (x0, y0)
        with a given angle (in degrees).

        Parameters:
        - x, y, z: 2D numpy arrays defining the mesh grids.
        - x0, y0: The point through which the slice passes.
        - angle_deg: The angle of the slice in degrees (0-360).
        - num_points: Number of points along the slice.
        - t_range: The range of the parameter t (distance from (x0, y0)) to consider.

        Returns:
        - t: Array of distances along the slice from (x0, y0).
        - z_slice: Interpolated z values along the slice.
        """
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)

        # Create direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Create parameter t (distance along the line)
        t = np.linspace(-t_range, t_range, num_points)

        # Compute points along the line
        x_line = x0 + t * dx
        y_line = y0 + t * dy

        # Create interpolator for z
        interpolator = RegularGridInterpolator(
            (y[:, 0], x[0, :]), z, bounds_error=False, fill_value=None
        )

        # Interpolate z values
        points = np.column_stack((x_line.ravel(), y_line.ravel()))
        z_slice = interpolator(points).reshape(x_line.shape)

        return t, z_slice

    return (slice_3d_surface,)


@app.cell
def _(go):
    def add_custom_dashed_line_3d(fig, x, y, z_min, z_max,
                                   pattern=(0.6, 0.1, 0.1, 0.1),
                                   color='black', width=3):
        """
        Adds a vertical dashed line in 3D with a custom dash pattern.

        pattern: tuple of (draw, gap, draw, gap, ...) in absolute units
                 e.g. (0.6, 0.1, 0.1, 0.1) means:
                 - draw 0.6, gap 0.1, draw 0.1, gap 0.1  (repeating)
        """
        z_segments = []
        x_segments = []
        y_segments = []

        z_current = z_min
        idx = 0       # index in pattern
        draw = True   # start with drawing

        while z_current < z_max:
            segment_length = pattern[idx % len(pattern)]
            z_next = min(z_current + segment_length, z_max)

            if draw:
                z_segments += [z_current, z_next, None]
                x_segments += [x, x, None]
                y_segments += [y, y, None]

            z_current = z_next
            idx += 1
            draw = not draw  # alternate draw/gap

        fig.add_trace(go.Scatter3d(
            x=x_segments,
            y=y_segments,
            z=z_segments,
            mode='lines',
            line=dict(color=color, width=width),
            showlegend=False,
            name='Custom Dashed Axis'
        ))

        return fig

    return (add_custom_dashed_line_3d,)


@app.cell
def _(np):
    def eval_asphere_raw(r, R, k, asph_coeffs):
        return r**2 / R / (1+np.sqrt(1-(1+k)*r**2/R**2)) + np.sum([asph_coeffs[i] * r**(2*(i+2)) for i in range(len(asph_coeffs))], axis=0)

    def eval_asphere(x, y, R, k, asph_coeffs, diam_opt, diam_ext):
        r = np.sqrt(x**2+y**2)
        asphere_raw = eval_asphere_raw(r, R, k, asph_coeffs)
        asphere_raw[(r > diam_opt/2) & (r <= diam_ext/2)] = eval_asphere_raw(diam_opt/2, R, k, asph_coeffs)
        asphere_raw[r > diam_ext/2] = np.nan

        return asphere_raw


    return eval_asphere, eval_asphere_raw


@app.cell
def _(np):
    R = 20
    k = 0.0
    asph_coeffs = []
    diam_opt = 16
    diam_ext = 20

    _L, _K = 200, 250
    _ddx = np.linspace(-10.0, 10.0, _K)
    _ddy = np.linspace(-10.0, 10.0, _L)
    xv, yv = np.meshgrid(_ddx, _ddy)
    return R, asph_coeffs, diam_ext, diam_opt, k, xv, yv


@app.cell
def _(
    R,
    add_custom_dashed_line_3d,
    asph_coeffs,
    diam_ext,
    diam_opt,
    eval_asphere,
    go,
    k,
    mo,
    np,
    slice_3d_surface,
    xv,
    yv,
):
    zv = eval_asphere(xv, yv, R, k, asph_coeffs, diam_opt, diam_ext)
    _p = go.Figure()
    _p.add_trace(
        go.Surface(
            x=xv,
            y=yv,
            z=zv,
            contours_z=dict(
                show=True,
                # usecolormap=True,
                highlightcolor="limegreen",
                # project_z=True,
                start=-1.5,
                end=1.5,
                size=0.4,
                color="white"
            ),
            showscale=False,
            # cmin=-1.5,
            # cmax=1.5,
        )
    )
    _p.update_layout(
        width=800,  # Set width (in pixels)
        height=800,  # Set height (in pixels)
        margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
        scene=dict(
            domain=dict(x=[0, 1], y=[0, 1]),  # Make the 3D scene fill the entire figure
            xaxis=dict(
                nticks=4,
                range=[-10.0, 10.0],
            ),
            yaxis=dict(
                nticks=4,
                range=[-10.0, 10.0],
            ),
            zaxis=dict(
                nticks=4,
                range=[-3, 3.5],
            ),
        ),
    )
    _p.update_scenes(
        camera=dict(
            eye=dict(x=1., y=1.5, z=0.8),  # Camera position (higher = farther away)
            up=dict(x=0, y=0, z=1),  # "Up" direction (default: z-axis)
            center=dict(x=0, y=0, z=0),  # Point the camera is looking at
        ),
        aspectratio={"x": 1, "y": 1, "z": 0.5},
        camera_projection_type="orthographic",
    )

    slice_x, slice_z = slice_3d_surface(xv, yv, zv, 0, 0, 0, num_points=100, t_range=10.0)
    slice_y = np.zeros_like(slice_x)


    _p.add_trace(
        go.Scatter3d(
            x=slice_x,
            y=slice_y,
            z=slice_z,
            mode="lines",
            line=dict(
                color='red',
                width=10

            ),showlegend=False
        )
    )

    _p = add_custom_dashed_line_3d(_p, x=0, y=0, z_min=-3, z_max=3.5, pattern=(0.6, 0.15, 0.1, 0.15), width=5, color="white")


    # Create a grid for the plane (adjust resolution as needed)
    _x = np.linspace(-10, 10, 5) 
    _z = np.linspace(-3, 3.5, 5)
    _x_grid, _z_grid = np.meshgrid(_x, _z)

    # Create x-grid (constant for vertical plane)
    _y_grid = np.full_like(_x_grid, 0.0)


    # Add the plane to the figure
    _p.add_trace(
        go.Surface(
            x=_x_grid,
            y=_y_grid,
            z=_z_grid,
            opacity=0.3,                        # Semi-transparency (0=invisible, 1=solid)
        colorscale=[[0, 'blue'], [1, 'blue']],
            showscale=False,  # Hide color scale
            name="Vertical Plane",
        )
    )
    _plot = mo.ui.plotly(_p)
    mo.vstack([_plot])
    return


@app.cell
def _(R, eval_asphere_raw, go, k, np):
    _asph_coeffs = [0, -0.000001]
    _r = np.linspace(-10, 10, 101)
    _asph_nominal = eval_asphere_raw(_r, R, k, _asph_coeffs)
    _asph_min = eval_asphere_raw(_r, R+5, k, _asph_coeffs)
    _asph_max = eval_asphere_raw(_r, R-5, k, _asph_coeffs)
    _asph_measured = eval_asphere_raw(_r, R+3, k, _asph_coeffs)
    _asph_measured_w_noise = _asph_measured + np.random.normal(0,0.05,101)
    _p = go.Figure()
    _p.add_trace(
        go.Line(
            x=_r,
            y=_asph_nominal,
            line=dict(
                color="royalblue",
                width=2,
                dash="dash",
            ),
            name="Nominal profile"
        )
    )
    _p.add_trace(
        go.Line(
            x=_r,
            y=_asph_min,
            line=dict(
                color="royalblue",
                width=0.5
            ),
            showlegend=False
        )
    )
    _p.add_trace(
        go.Line(
            x=_r,
            y=_asph_max,
            line=dict(
                color="royalblue",
                width=0.5
            ),
            fill="tonexty",
            fillcolor="rgba(166, 184, 255, 0.2)",
            showlegend=False
        )
    )
    _p.add_trace(
        go.Line(
            x=_r,
            y=_asph_measured_w_noise,
            line=dict(
                color="darkgreen",
                width=1,
            ),
            name="Measured profile"
        )
    )
    _p.add_trace(
        go.Line(
            x=_r,
            y=_asph_measured,
            line=dict(
                color="darkgreen",
                width=2,
                # dash="dash"
            ),
            name="Nominal profile w/ optimized radius"
        )
    )
    _p.update_layout(
        width=800,  # Set width (in pixels)
        height=400,  # Set height (in pixels)
    )
    config = {
      'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'optimized_radius',
        'height': 400,
        'width': 800,
        'scale': 6
      }
    }
    # _p.update_layout(template="simple_white")
    _p.show(config=config)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

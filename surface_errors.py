import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/surface_errors.slides.json",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Form error visualizer
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.interpolate import RegularGridInterpolator
    from zernike import RZern

    return RZern, RegularGridInterpolator, go, mo, np, px


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
def _(RZern, np):
    cart = RZern(6)
    L, K = 200, 250
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    return cart, xv, yv


@app.cell
def _():
    # _c = np.zeros(cart.nk)
    # plt.figure()
    # for i in range(1, 10):
    #     plt.subplot(3, 3, i)
    #     _c *= 0.0
    #     _c[i] = 1.0
    #     _Phi = cart.eval_grid(_c, matrix=True)
    #     plt.imshow(_Phi, origin='lower', extent=(-1, 1, -1, 1))
    #     plt.axis('off')
    # plt.show()
    return


@app.cell
def _(mo):
    ast_amp_slider = mo.ui.slider(
        label="Ast amplitude",
        start=-0.4,
        stop=0.4,
        step=0.05,
        value=0.0,
        debounce=False,
    )
    ast_azimuth_slider = mo.ui.slider(
        label="Ast azimuth", start=-180, stop=180.0, step=1, value=0.0, debounce=False
    )
    rsi_slider = mo.ui.slider(
        label="RSI", start=-0.3, stop=0.3, step=0.02, value=0.0, debounce=False
    )
    power_slider = mo.ui.slider(
        label="Power", start=-0.5, stop=0.5, step=0.05, value=0.0, debounce=False
    )
    return ast_amp_slider, ast_azimuth_slider, power_slider, rsi_slider


@app.cell
def _(
    ast_amp_slider,
    ast_azimuth_slider,
    cart,
    go,
    mo,
    np,
    power_slider,
    rsi_slider,
    slice_3d_surface,
    xv,
    yv,
):
    _c = np.zeros(cart.nk)
    _theta = 2 * np.radians(ast_azimuth_slider.value)
    _amp = ast_amp_slider.value
    _c[3] = power_slider.value + 3 / 4 * np.sqrt(5.0 / 6) * rsi_slider.value
    _c[4] = _amp * (np.cos(_theta) - np.sin(_theta))
    _c[5] = _amp * (np.sin(_theta) + np.cos(_theta))
    _c[10] = rsi_slider.value

    Phi = cart.eval_grid(_c, matrix=True) + np.sqrt(3) * _c[3] - np.sqrt(5) * _c[10]
    _p = go.Figure()
    _p.add_trace(
        go.Surface(
            x=xv,
            y=yv,
            z=Phi,
            contours_z=dict(
                    show=True,
                    # usecolormap=True,
                    highlightcolor="limegreen",
                    # project_z=True,
                    start=-2.5,
                    end=2.5,
                    size=0.15,
                    color="white"
                ),
            showscale=False,
            cmin=-1.5,
            cmax=1.5,
        )
    )
    _p.update_layout(
        width=600,  # Set width (in pixels)
        height=600,  # Set height (in pixels)
        margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
        scene=dict(
            domain=dict(x=[0, 1], y=[0, 1]),  # Make the 3D scene fill the entire figure
            xaxis=dict(
                nticks=4,
                range=[-1.0, 1.0],
            ),
            yaxis=dict(
                nticks=4,
                range=[-1.0, 1.0],
            ),
            zaxis=dict(
                nticks=4,
                range=[-2.5, 2.5],
            ),
        ),
    )
    _p.update_scenes(
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8),  # Camera position (higher = farther away)
            up=dict(x=0, y=0, z=1),  # "Up" direction (default: z-axis)
            center=dict(x=0, y=0, z=0),  # Point the camera is looking at
        ),
        aspectratio={"x": 1, "y": 1, "z": 0.5},
        camera_projection_type="orthographic",
    )
    _plot = mo.ui.plotly(_p)
    # plt.imshow(_Phi, origin='lower', extent=(-1, 1, -1, 1))
    # print(np.nanmax(_Phi))

    _fig = go.Figure()
    _angles = np.linspace(0, 180, 8, endpoint=False)
    _z_slices = []
    for _angle in _angles:
        _t, _z_slice = slice_3d_surface(
            xv, yv, Phi, x0=0.0, y0=0.0, angle_deg=_angle, t_range=1
        )
        _z_slices.append(_z_slice)
        _fig.add_trace(
            go.Scatter(
                x=_t, y=_z_slice, mode="lines", line={"color": "royalblue", "width": 1}
            )
        )
    _fig.add_trace(
        go.Scatter(
            x=_t, y=np.mean(_z_slices, axis=0), mode="lines", line={"color": "crimson"}
        )
    )

    _fig.update_xaxes(range=[-1, 1])
    _fig.update_yaxes(range=[-2, 2])
    _fig.update_layout(
        showlegend=False,
        width=450,  # Set width (in pixels)
        height=400,  # Set height (in pixels)
        margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
    )
    _plot_lines = mo.ui.plotly(_fig)

    _sliders = mo.vstack(
        [ast_amp_slider, ast_azimuth_slider, rsi_slider, power_slider], gap=1
    )
    mo.vstack(
        [mo.hstack([_plot, _plot_lines], gap=0, align="center"), _sliders],
        justify="start",
        align="center",
        gap=1,
    )
    return (Phi,)


@app.cell
def _(mo):
    slice_x_slider = mo.ui.slider(
        label="Slice x0", start=-1.0, stop=1.0, step=0.1, value=0.0, debounce=False
    )
    slice_y_slider = mo.ui.slider(
        label="Slice y0", start=-1.0, stop=1.0, step=0.1, value=0.0, debounce=False
    )
    slice_angle_slider = mo.ui.slider(
        label="Slice angle", start=-180, stop=180, step=0.1, value=0.0, debounce=False
    )
    return slice_angle_slider, slice_x_slider, slice_y_slider


@app.cell
def _(
    Phi,
    mo,
    px,
    slice_3d_surface,
    slice_angle_slider,
    slice_x_slider,
    slice_y_slider,
    xv,
    yv,
):
    t, z_slice = slice_3d_surface(
        xv,
        yv,
        Phi,
        x0=slice_x_slider.value,
        y0=slice_y_slider.value,
        angle_deg=slice_angle_slider.value,
        t_range=1,
    )
    _fig = px.line(x=t, y=z_slice, width=600, height=300)
    _fig.update_xaxes(range=[-1, 1])
    _fig.update_yaxes(range=[-1, 1])
    _plot = mo.ui.plotly(_fig)
    _sliders = mo.vstack([slice_x_slider, slice_y_slider, slice_angle_slider], gap=1)
    mo.hstack([_plot, _sliders], justify="start", align="center", gap=1)
    return


@app.cell
def _(cart, go, mo, np, xv, yv):
    def plot_surface(theta, ast_amp, power_amp, rsi_amp):
        _c = np.zeros(cart.nk)
        _theta = 2 * np.radians(theta)
        _amp = ast_amp
        _c[3] = power_amp + 3 / 4 * np.sqrt(5.0 / 6) * rsi_amp
        _c[4] = _amp * (np.cos(_theta) - np.sin(_theta))
        _c[5] = _amp * (np.sin(_theta) + np.cos(_theta))
        _c[10] = rsi_amp

        _Phi = cart.eval_grid(_c, matrix=True) + np.sqrt(3) * _c[3] - np.sqrt(5) * _c[10]
        _p = go.Figure()
        _p.add_trace(
            go.Surface(
                x=xv,
                y=yv,
                z=_Phi,
                contours_z=dict(
                    show=True,
                    # usecolormap=True,
                    highlightcolor="limegreen",
                    # project_z=True,
                    start=-1.5,
                    end=1.5,
                    size=0.15,
                    color="white"
                ),
                showscale=False,
                cmin=-1.5,
                cmax=1.5,
            )
        )
        _p.update_layout(
            width=600,  # Set width (in pixels)
            height=600,  # Set height (in pixels)
            margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
            scene=dict(
                domain=dict(x=[0, 1], y=[0, 1]),  # Make the 3D scene fill the entire figure
                xaxis=dict(
                    nticks=4,
                    range=[-1.0, 1.0],
                ),
                yaxis=dict(
                    nticks=4,
                    range=[-1.0, 1.0],
                ),
                zaxis=dict(
                    nticks=4,
                    range=[-2.5, 2.5],
                ),
            ),
        )
        _p.update_scenes(
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),  # Camera position (higher = farther away)
                up=dict(x=0, y=0, z=1),  # "Up" direction (default: z-axis)
                center=dict(x=0, y=0, z=0),  # Point the camera is looking at
            ),
            aspectratio={"x": 1, "y": 1, "z": 0.5},
            camera_projection_type="orthographic",
        )
        _plot = mo.ui.plotly(_p)
        # plt.imshow(_Phi, origin='lower', extent=(-1, 1, -1, 1))
        # print(np.nanmax(_Phi))

        return _plot

    return


@app.cell
def _():
    # mo.vstack([plot_surface(0, 0.15, 0.35, 0.15),
    # plot_surface(0, 0., 0.35, 0.),
    # plot_surface(0, 0.15, 0.0, 0.15),
    # plot_surface(0, 0., 0.0, 0.15),
    # plot_surface(0, 0.15, 0.0, 0.),])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

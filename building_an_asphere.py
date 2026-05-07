import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib

    font = {"family": "Arial", "weight": "normal", "size": 10}

    matplotlib.rc("font", **font)

    import matplotlib.pyplot as plt
    import numpy as np


    return np, plt


@app.cell
def _(np):
    R = 10
    k = 0
    asph_coeffs = [-0.05, 0.007]
    x = np.linspace(-1, 1, 100)
    y_sphere = x**2 / R / (1+np.sqrt(1-(1+k)*x**2/R**2))
    y_asphere = np.sum([asph_coeffs[i] * x**(2*(i+2)) for i in range(len(asph_coeffs))], axis=0)
    return asph_coeffs, x, y_asphere, y_sphere


@app.cell
def _(asph_coeffs, x):
    [asph_coeffs[i] * x**(2*(i+2)) for i in range(len(asph_coeffs))]
    return


@app.cell
def _(plt, x, y_asphere, y_sphere):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y_sphere, label='Sphere + conic')
    plt.plot(x, y_asphere, label='Polynomial')
    plt.plot(x, y_sphere + y_asphere, label='Combined')
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    xlims = [-1, 1]
    ylims = [-0.05, 0.05]
    return xlims, ylims


@app.cell
def _(plt, x, xlims, y_sphere, ylims):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y_sphere, color="#4a8222")
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Horizontal line at y=0
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)  # Vertical line at x=0
    plt.tick_params(
        axis='both',       # Apply to both x and y axes
        which='both',      # Major and minor ticks
        bottom=False,      # Remove ticks on the bottom
        top=False,         # Remove ticks on the top
        left=False,        # Remove ticks on the left
        right=False,       # Remove ticks on the right
        labelbottom=False, # Remove x-axis labels
        labelleft=False    # Remove y-axis labels
    )
    # plt.grid()
    plt.show()
    return


@app.cell
def _(plt, x, xlims, y_asphere, ylims):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y_asphere, color="#23a2c6")
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.axhline(y=0, color='0.6', linestyle='-', linewidth=0.5)  # Horizontal line at y=0
    plt.axvline(x=0, color='0.6', linestyle='-', linewidth=0.5)  # Vertical line at x=0
    plt.tick_params(
        axis='both',       # Apply to both x and y axes
        which='both',      # Major and minor ticks
        bottom=False,      # Remove ticks on the bottom
        top=False,         # Remove ticks on the top
        left=False,        # Remove ticks on the left
        right=False,       # Remove ticks on the right
        labelbottom=False, # Remove x-axis labels
        labelleft=False    # Remove y-axis labels
    )
    # plt.grid()
    plt.show()
    return


@app.cell
def _(plt, x, xlims, y_asphere, y_sphere, ylims):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y_sphere + y_asphere, "C1")
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.axhline(y=0, color='0.6', linestyle='-', linewidth=0.5)  # Horizontal line at y=0
    plt.axvline(x=0, color='0.6', linestyle='-', linewidth=0.5)  # Vertical line at x=0
    plt.tick_params(
        axis='both',       # Apply to both x and y axes
        which='both',      # Major and minor ticks
        bottom=False,      # Remove ticks on the bottom
        top=False,         # Remove ticks on the top
        left=False,        # Remove ticks on the left
        right=False,       # Remove ticks on the right
        labelbottom=False, # Remove x-axis labels
        labelleft=False    # Remove y-axis labels
    )
    # plt.grid()
    plt.show()
    return


@app.cell
def _(plt, x, xlims, y_asphere, y_sphere):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y_sphere + y_asphere, "--C1")
    plt.plot(x, y_sphere + y_asphere + 0.015*x**2-0.005*x**4, "-C1")
    plt.xlim(*xlims)
    plt.ylim(-0.02, 0.04)
    plt.axhline(y=0, color='0.6', linestyle='-', linewidth=0.5)  # Horizontal line at y=0
    plt.axvline(x=0, color='0.6', linestyle='-', linewidth=0.5)  # Vertical line at x=0
    plt.tick_params(
        axis='both',       # Apply to both x and y axes
        which='both',      # Major and minor ticks
        bottom=False,      # Remove ticks on the bottom
        top=False,         # Remove ticks on the top
        left=False,        # Remove ticks on the left
        right=False,       # Remove ticks on the right
        labelbottom=False, # Remove x-axis labels
        labelleft=False    # Remove y-axis labels
    )
    # plt.grid()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import matplotlib

    font = {"family": "Arial", "weight": "normal", "size": 12}

    matplotlib.rc("font", **font)

    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo


    return np, plt


@app.cell
def _(np, plt):
    _x = np.linspace(-10, 10, 101)
    _R = 15
    _k = 0
    _asph_coeffs = [0.0004, -0.000004]
    _y = _x**2 / _R / (1+np.sqrt(1-(1+_k)*_x**2/_R**2)) + np.sum([_asph_coeffs[_i] * _x**(2*(_i+2)) for _i in range(len(_asph_coeffs))], axis=0)

    x_alt = _x[2::16]
    y_alt = _y[2::16]
    plt.figure(figsize=(7, 5))
    plt.plot(_x, _y, "-C1")
    plt.scatter(x=x_alt, y=y_alt, color="C1")
    for i, (_x_i, _y_i) in enumerate(zip(x_alt, y_alt), start=-3):
        plt.plot((0, _x_i), (_y_i, _y_i), "--C1", linewidth=0.5)
        if i < 0:
            pass
        else:
            plt.text(_x_i/2, _y_i, f"$r_{{{i}}}$", va="bottom")
        plt.plot((_x_i, _x_i), (0, _y_i), "--C1", linewidth=0.5)
        if i < 0:
            # plt.text(_x_i, _y_i/2, f"$z_{{{i}}}$", ha="left", va="top")
            pass
        else:
            plt.text(_x_i, _y_i/2, f"$z_{{{i}}}$", ha="right", va="top")
    plt.xlim(-10, 10)
    plt.ylim(-1, 5)
    plt.axhline(y=0, color='0.6', linestyle='-', linewidth=0.5)  # Horizontal line at y=0
    plt.axvline(x=0, color='0.6', linestyle='-', linewidth=0.5)  # Vertical line at x=0
    # plt.tick_params(
    #     axis='both',       # Apply to both x and y axes
    #     which='both',      # Major and minor ticks
    #     bottom=False,      # Remove ticks on the bottom
    #     top=False,         # Remove ticks on the top
    #     left=False,        # Remove ticks on the left
    #     right=False,       # Remove ticks on the right
    #     labelbottom=False, # Remove x-axis labels
    #     labelleft=False    # Remove y-axis labels
    # )
    # plt.grid()
    plt.xlabel("Radius position")
    plt.ylabel("Base surface sag")
    plt.show()
    return x_alt, y_alt


@app.cell
def _(x_alt, y_alt):
    sequence = [["Radius position", "Base surface sag"]]
    for _x_alt_i, _y_alt_i in zip(x_alt, y_alt):
        sequence.append([f"{_x_alt_i:.6f}", f"{_y_alt_i:.6f}"])
    rows = []
    for row in sequence:
        rows.append(f"{row[0]:>15}   {row[1]}")
    print('\n'.join(rows))
    return (sequence,)


@app.cell
def _(sequence):
    sequence
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

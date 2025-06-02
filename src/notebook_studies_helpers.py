# Helpers for the notebook studies (e.g. convergence studies).

import numpy as np
import matplotlib.pyplot as plt
import nbformat as nbf

from typing import List, Tuple

from utils_for_testing import observed_rates_report
import prob1base as p1


def report_on_rates(
    errors: list[float],
    *,
    expected_rate: float = 2.0,
    tolerance: float = 0.1,
    cmp_type: str = "least",  # 'equal' or 'least' for expected rate = target, versus >= target
    title: str = "Observed Rates",
) -> Tuple[List[float], bool]:
    """
    Wrapper for observed_rates_report that doesn't assert on bad rates.

    Args:
        errors: A list of error values obtained at different refinement levels
        expected_rate: The expected convergence rate
        tolerance: Tolerance for comparing observed and expected rates
        cmp_type: Comparison type ('equal' or 'least')
        title: Title for the rate report

    Returns:
        Tuple containing list of observed rates and a boolean indicating success
    """
    print(f"\n{title}:")
    print("-" * len(title))

    try:
        observed_rates = observed_rates_report(
            errors=errors,
            expected_rate=expected_rate,
            tolerance=tolerance,
            cmp_type=cmp_type,
            halt_print=False,  # We want to see the detailed output
        )
        success = True
    except Exception as e:
        print(f"❌ Error in rate calculation: {e}")
        print(
            f"❌ Does not match expected rate of {expected_rate:.1f} (within tolerance ±{tolerance:.1f})"
        )
        return [], False

    final_rate = observed_rates[-1]
    print(f"Final rate: {final_rate:.3f}")
    print(
        f"✅ Matches expected rate of {expected_rate:.1f} (within tolerance ±{tolerance:.1f})"
    )

    return observed_rates, success


def plot_errors_and_rates(
    errors: List[float],
    rates: List[float],
    x_values: List,
    x_label: str,
    title: str,
    expected_rate: float = None,
    log_scale: bool = True,
):
    """
    Plot the errors and rates from a convergence study.

    Args:
        errors: List of error values
        rates: List of calculated convergence rates
        x_values: List of x-axis values (grid sizes or time steps)
        x_label: Label for x-axis
        title: Plot title
        expected_rate: Expected convergence rate (for reference line)
        log_scale: Whether to use log scale for the plots
    """
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Error plot
    ax1.plot(x_values, errors, "o-", linewidth=2, markersize=8)
    if log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Error")
    ax1.set_title(f"{title} - Error Values")
    ax1.grid(True, which="both", ls="-")

    # Rate plot
    if rates:
        rate_x_values = x_values[1:-1]  # Rates are calculated between error points
        ax2.plot(rate_x_values, rates, "o-", linewidth=2, markersize=8)
        if expected_rate:
            ax2.axhline(
                y=expected_rate,
                color="r",
                linestyle="--",
                label=f"Expected Rate = {expected_rate}",
            )
            ax2.legend()
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("Convergence Rate")
        ax2.set_title(f"{title} - Convergence Rates")
        ax2.grid(True)
    else:
        ax2.text(
            0.5,
            0.5,
            "Not enough data points\nto calculate rates",
            ha="center",
            va="center",
            fontsize=14,
        )

    plt.tight_layout()
    plt.show()


def visualize_mms_solution(mms_case, time_point=0.1):
    """
    Visualize the MMS solution at a specific time point

    Args:
        mms_case: The MMS case object
        time_point: Time point for visualization
    """
    grid = mms_case.grid
    xx, yy = grid.xx, grid.yy

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Get solutions at the given time point
    T = mms_case.T(time_point, xx, yy)
    cp = mms_case.cp(time_point, xx, yy)
    cl = mms_case.cl(time_point, xx, yy)
    cd = mms_case.cd(time_point, xx, yy)
    cs = mms_case.cs(time_point, xx, yy)

    # Plot each variable
    im1 = axs[0, 0].contourf(xx, yy, T, cmap="hot")
    plt.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].set_title("T (Temperature)")

    im2 = axs[0, 1].contourf(xx, yy, cp, cmap="viridis")
    plt.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].set_title("cp (Product Concentration)")

    im3 = axs[0, 2].contourf(xx, yy, cl, cmap="Blues")
    plt.colorbar(im3, ax=axs[0, 2])
    axs[0, 2].set_title("cl (Liquid Concentration)")

    im4 = axs[1, 0].contourf(xx, yy, cd, cmap="Greens")
    plt.colorbar(im4, ax=axs[1, 0])
    axs[1, 0].set_title("cd (Dissolved Concentration)")

    im5 = axs[1, 1].contourf(xx, yy, cs, cmap="Purples")
    plt.colorbar(im5, ax=axs[1, 1])
    axs[1, 1].set_title("cs (Solid Concentration)")

    # Show the regularized Heaviside function for reference
    x_vals = np.linspace(-1, 1, 1000)
    for reg_factor in [10, 50, 100]:
        h_vals = p1.heaviside_regularized(x_vals, reg_factor)
        axs[1, 2].plot(x_vals, h_vals, label=f"η = {reg_factor}")

    axs[1, 2].set_title("Regularized Heaviside Function")
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.suptitle(f"MMS Solution at time t = {time_point}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    plt.show()


# I've been copying/pasting notebook files to create new ones, leading to code repetition, misaligned formatting, etc. I should have a base template and use it to generate new notebooks with this funcition. Essentially I'll need to modify which MMS Case goes into each notebook, and that's it.
def modify_notebook_cell(
    template_path: str, output_path: str, cell_index: int, new_cell_content: str
):
    """
    Loads a template notebook, modifies a specific cell's content,
    and saves it as a new notebook.

    Args:
        template_path (str): Path to the input .ipynb template file.
        output_path (str): Path to save the modified .ipynb file.
        cell_index (int): The 0-based index of the cell to modify.
        new_cell_content (str): The new Python code string for the cell.
    """

    # This code needs proper reviewing and understanding. It's AI generated about something I never used before (nbformat). I've performed an initial review and polish, but there are still some questions and uncertainties.

    # 1. Read the template notebook
    # We use `nbf.NO_CONVERT` to preserve the original notebook version
    # unless a specific version is required.
    notebook = nbf.read(template_path, as_version=nbf.NO_CONVERT)
    # NOTE: Do I need this above about nfg.NO_CONVERT? This code is AI suggested. I'm not sure if it is necessary.

    assert 0 <= cell_index < len(notebook.cells), "Cell index out of bounds."
    # Does this indexing operator already performs bounds "checking" (through assertions) or does it create a new cell if the index is out of bounds? More specifically, do I need the above assertion?
    notebook.cells[cell_index].source = new_cell_content

    # Isn't the utf-8 encoding the default?
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)

import math

import numpy as np
from cvg_studies_base import calculate_observed_rates, RateStatus
from typing import List, Literal

def solve_newton_step_Fx_eq_C(*, x0, Fx0, JacFx0, C):
    """
    Performs one Newton step for solving the system F(x) = C.

    Given the current iterate x0, the function value F(x0), the Jacobian
    JacFx0 = dF/dx evaluated at x0, and the target vector C, this function
    calculates the next iterate x1 using the formula:
        x1 = x0 + JacFx0^{-1} * (C - F(x0))

    Args:
        x0 (np.ndarray): Current iterate vector (1D).
        Fx0 (np.ndarray): Function value F(x) evaluated at x0 (1D).
        JacFx0 (np.ndarray): Jacobian matrix dF/dx evaluated at x0 (2D).
                           Assumed to be dense for np.linalg.solve.
        C (np.ndarray): Target constant vector (1D).

    Returns:
        np.ndarray: The next iterate vector x1 (1D).

    Raises:
        ValueError: If input dimensions are inconsistent.
        np.linalg.LinAlgError: If the Jacobian matrix is singular.
    """
    # --- Input Validation ---
    if not isinstance(x0, np.ndarray) or x0.ndim != 1:
        raise ValueError("x0 must be a 1D NumPy array.")
    if not isinstance(Fx0, np.ndarray) or Fx0.ndim != 1:
        raise ValueError("Fx0 must be a 1D NumPy array.")
    if not isinstance(C, np.ndarray) or C.ndim != 1:
        raise ValueError("C must be a 1D NumPy array.")
    if not isinstance(JacFx0, np.ndarray) or JacFx0.ndim != 2:
        raise ValueError("JacFx0 must be a 2D NumPy array.")

    dim = x0.shape[0]
    if Fx0.shape != (dim,) or C.shape != (dim,) or JacFx0.shape != (dim, dim):
        raise ValueError(
            f"Inconsistent dimensions: x0({x0.shape}), Fx0({Fx0.shape}), "
            f"C({C.shape}), JacFx0({JacFx0.shape})"
        )

    # --- Newton Step Calculation ---
    # 1. Calculate the residual vector (right-hand side for the linear solve)
    residual = C - Fx0

    # 2. Solve the linear system JacFx0 * delta_x = residual for the update delta_x
    #    np.linalg.solve is suitable for dense matrices.
    try:
        delta_x = np.linalg.solve(JacFx0, residual)
    except np.linalg.LinAlgError as e:
        # Re-raise error if Jacobian is singular
        raise np.linalg.LinAlgError(f"Jacobian matrix is singular: {e}")

    # 3. Calculate the next iterate
    x1 = x0 + delta_x

    return x1

# NOTE: Has significant duplication with cvg_studies_base.py's calculate_observed_rates. Needs to fix this.
def observed_rates_report(
    errors: list[float],
    *,
    expected_rate: float = 2.0,
    tolerance: float = 0.1,
    cmp_type: str = "least",  # 'equal' or 'least' for expected rate = target, verus >= target
    halt_print: bool = False,
) -> list:
    """
    Calculates observed convergence rates using a 3-point formula and reports them.

    Args:
        errors (list): A list of error values obtained at different refinement levels.
        expected_rate (float): The expected convergence rate (default: 2.0).
        tolerance (float): The tolerance for comparing the observed rate with the expected rate (default: 0.1).
        cmp_type (str): Comparison type for expected rate. 'equal' for observed rate == expected rate, 'least' for observed rate >= expected rate - tolerance.
        halt_print (bool): If True, this function won't print anything and just return the observed rates. If False, it'll print the found rates. False by default.

    Returns:
        list: A list of observed convergence rates.
    """

    effectively_zero_tol = 1e-15

    if cmp_type not in ["equal", "least"]:
        raise ValueError(f"cmp_type must be 'equal' or 'least', not {cmp_type}")

    def cond_print(*args):
        if not halt_print:
            print(*args)

    refinements_steps = len(errors)
    observed_rates = []
    cond_print("\nObserved Rates (3-point formula):")
    for k in range(refinements_steps - 2):
        err_coarse = errors[k]
        err_medium = errors[k + 1]
        err_fine = errors[k + 2]

        rate = float("nan")
        numerator = err_coarse - err_medium
        denominator = err_medium - err_fine

        too_small = 1e-16

        if denominator > too_small and numerator > too_small:
            ratio = numerator / denominator
            if ratio > 0:
                rate = math.log2(ratio)
            else:
                cond_print(
                    f"    Warning: Ratio ({ratio:.2e}) non-positive: levels {k},{k+1},{k+2}."
                )
        elif abs(denominator) <= too_small:
            if abs(numerator) <= too_small:
                cond_print(
                    f"    Note: Differences zero: levels {k},{k+1},{k+2} (error={err_fine:.2e})."
                )
            else:
                cond_print(
                    f"    Warning: Denom near zero ({denominator:.2e}), num non-zero ({numerator:.2e}): levels {k},{k+1},{k+2}."
                )
        else:
            cond_print(
                f"    Warning: Numerator non-positive ({numerator:.2e}): levels {k},{k+1},{k+2}."
            )

        observed_rates.append(rate)
        if np.isfinite(rate):
            cond_print(
                f"    Levels {k},{k+1},{k+2}: log2({numerator:.3e} / {denominator:.3e}) = {rate:.3f}"
            )
        else:
            cond_print(
                f"    Levels {k},{k+1},{k+2}: log2({numerator:.3e} / {denominator:.3e}) = NaN"
            )

    # --- Assertions ---
    cmp_symbol = '≅' if cmp_type == "equal" else '⪆' 
    cond_print(f"\nAssertion (expecting {cmp_symbol}{expected_rate}):")
    assert len(observed_rates) > 0, "Not enough refinement levels."
    final_rate = observed_rates[-1]
    
    are_there_effectively_zero_errors = any(abs(err) < effectively_zero_tol for err in errors)
    is_finite_final_rate = np.isfinite(final_rate)
    if  are_there_effectively_zero_errors and not is_finite_final_rate:
        cond_print("Non-finite final rate, with at least one effectively zero error present. Not failing.")
        return observed_rates
    assert is_finite_final_rate, f"Final rate is not finite ({final_rate})."

    cond_print(f"  Final observed rate (3-point): {final_rate:.3f}")

    if cmp_type == "least":
        assert (
            final_rate >= expected_rate - tolerance
        ), f"Observed rate {final_rate:.3f} not at least {expected_rate:.1f}"

    else:
        assert np.isclose(
            final_rate, expected_rate, atol=tolerance
        ), f"Observed rate {final_rate:.3f} not close to expected {expected_rate:.1f}"

    return observed_rates


# NOTE: This function depends on cvg_studies_base.py, which is currently obsolete in this project. This needs to be corrected. Some possible ways:
# 1. Use cvg_studies_base.py as a library for basic functions in which mms_trial_utils.py is implemented, and also this utils_for_testing.py. This is fine, but strange, I believe.
# 2. Move the functions from cvg_studies_base.py to mms_trial_utils.py, and then use them here. This is also fine as mms_trial_utils.py is the "new" convergence studies base, but properly designed for this project.
# 3. Move things like calculate_observed_rates and RateStatus here, as they are mostly to help testing this system, be it through pytest tests or through jupyter notebook style explorations.
def deliverable_from_test_caculate_observed_rates(
    *,
    name: str,
    errors: List[float],
    with_asserts: bool = True,
    target_order: float = 2.0,
    order_abs_tol: float = 0.1,
    cmp_type: Literal["least", "equal"] = "least",
) -> None:

    def assert_if_on(condition, message):
        if with_asserts:
            assert condition, message

    print(f"\n  Calculating observed rates for {name}...")

    valid_errors = [e for e in errors if e is not None and np.isfinite(e) and e >= 0]

    assert_if_on(
        len(valid_errors) >= 3,
        f"Insufficient valid {name} error points ({len(valid_errors)}) for rate calculation.",
    )

    rates_with_status = calculate_observed_rates(valid_errors, refinement_factor=2.0)

    assert_if_on(rates_with_status, f"Rate calculation for {name} failed.")

    # The idea is that your errors should have been calculated to reach asymptotic regime and, but with not too much refinement so that floating point issues get in the way. In such a case, the last rate should be the most accurate one.
    final_rate, final_status = rates_with_status[-1]

    print(f"  Rates({name}): {rates_with_status}")
    print(f"  Final Rate({name}): {final_rate}, Status: {final_status}")

    assert_if_on(
        final_status == RateStatus.OK,
        f"Rate calculation for {name} status: {final_status}",
    )

    match cmp_type:
        case "equal":
            assert_if_on(
                abs(final_rate - target_order) <= order_abs_tol,
                f"Expected spatial order {target_order} for {name}, but got {final_rate}",
            )
        case "least":
            assert_if_on(
                final_rate >= target_order - order_abs_tol,
                f"Expected spatial order at least {target_order} for {name}, but got {final_rate:.3f}",
            )
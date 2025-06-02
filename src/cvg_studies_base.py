# cvg_studies_base.py - v1

import math
import time
import numpy as np
import prob1base as p1
from collections import namedtuple
from typing import List, Tuple, Dict, Any, Literal, Optional, NamedTuple


# --- Rate Calculation Status Enum/Helper ---
# Using NamedTuple for simple structure, could be Enum
class _RateStatus(NamedTuple):
    OK: str = "OK"
    INSUFFICIENT_DATA: str = "Insufficient Data"
    ZERO_DENOMINATOR_ZERO_NUMERATOR: str = "Differences near zero (converged/stalled?)"
    ZERO_DENOMINATOR_NONZERO_NUMERATOR: str = "Unstable rate (denominator near zero)"
    NON_POSITIVE_RATIO: str = "Non-positive ratio (convergence issue?)"
    ERROR_INCREASING: str = "Error increasing significantly"


RateStatus = _RateStatus()


def calculate_observed_rates(
    errors: List[float], refinement_factor: float = 2.0
) -> List[Tuple[float, str]]:
    """
    Calculates observed convergence rates using a 3-point formula and returns status.

    Args:
        errors: A list of error values obtained at different refinement levels
                (assumed ordered from coarsest [index 0] to finest).
        refinement_factor: The factor by which the discretization parameter (h or dt)
                           is reduced between levels (default: 2.0).

    Returns:
        A list of tuples: [(rate_1, status_1), (rate_2, status_2), ...].
        'rate' is the calculated rate (float, can be nan/inf).
        'status' is a string indicating calculation status (e.g., from RateStatus).
    """
    results = []
    num_errors = len(errors)

    assert num_errors >= 3, "At least 3 error values are required for rate calculation."
    assert refinement_factor > 1.0, "Refinement factor must be > 1.0"

    log_r = math.log(refinement_factor)

    # Check if errors are all positive
    assert all(
        e >= 0 for e in errors
    ), "All error values must be positive for rate calculation."

    # Use a very small number to check for near-zero differences
    # Adjust based on expected error magnitudes if necessary
    near_zero_tol = np.finfo(float).eps

    # Iterate through triplets: (coarse, medium, fine)
    # Formula: E3-E2 / E2-E1 where 3=coarse, 1=fine.
    for k in range(num_errors - 2):
        err_coarse = errors[k]
        err_medium = errors[k + 1]
        err_fine = errors[k + 2]

        numerator = err_coarse - err_medium
        denominator = err_medium - err_fine

        rate = np.nan
        status = RateStatus.OK  # Default status

        if denominator < 0:  # Error increased significantly from medium to fine
            if numerator > 0:
                # Error decreased from coarse to medium then increased
                status = RateStatus.ERROR_INCREASING
            else:
                # Error increased consistently
                status = RateStatus.ERROR_INCREASING
            # rate remains nan
        elif numerator <= 0:
            # Error did not decrease from coarse to medium (or stayed same)
            # Denominator is positive (error decreased med->fine) but
            # numerator isn't.
            status = RateStatus.NON_POSITIVE_RATIO
            # rate remains nan
        else:
            # Standard case: numerator > 0, denominator > 0
            ratio = numerator / denominator

            if abs(denominator) < near_zero_tol:
                if abs(numerator) < near_zero_tol:
                    # 0/0 case, both numerator and denominator are near zero.
                    # This indicates convergence has stalled or is very close.
                    status = RateStatus.ZERO_DENOMINATOR_ZERO_NUMERATOR
                else:
                    # Denominator is zero, numerator is not. Unstable.
                    status = RateStatus.ZERO_DENOMINATOR_NONZERO_NUMERATOR

            # Check if ratio is positive (should be covered by above
            # checks, but defensive).
            assert ratio > 0
            rate = math.log(ratio) / log_r

        results.append((rate, status))

    return results


# --- Data structure for time series ---
TimeStepData = namedtuple(
    "TimeStepData", ["t", "h_norm_sq_errors", "grad_h_norm_p_sq_errors"]
)

# --- Helper Functions ---


def run_simulation_collect_data(
    *,
    grid: p1.Grid,
    integrator: p1.TimeIntegratorBase,
    exact_sol_pack: p1.MMSCaseBase,
    initial_state_gfp: p1.StateVars,
    Tf: float,
    dt: float,
    variable_names: List[str],
    integral_vars: List[str],
) -> Tuple[List[TimeStepData], float]:
    """
    Runs the simulation from t=0 to Tf, collecting necessary data at each step
    for the combined error norm calculation.
    """
    current_t = 0.0
    current_num_state = initial_state_gfp
    time_series_data = []

    num_steps = math.ceil(Tf / dt)
    dt = Tf / num_steps
    t_start_sim = time.time()
    # Want 10 prints for the entire simulation.
    steps_til_print = max(1, num_steps // 10)

    xx, yy = grid.xx, grid.yy

    def collect_errors(
        num_state: p1.StateVars, t: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Collects the H-norm and gradient H-norm errors at t=0.
        """
        exact_state_t = {v: getattr(exact_sol_pack, v)(t, xx, yy) for v in variable_names}

        # Initialize error dictionaries
        h_norm_sq_errors = {}
        grad_h_norm_p_sq_errors = {}

        # Calculate errors for each variable
        for var_name in variable_names:
            num_sol = getattr(num_state, var_name)
            exact_sol = exact_state_t[var_name]
            error = num_sol - exact_sol
            h_norm_sq_errors[var_name] = grid.norm_H(error) ** 2

            if var_name in integral_vars:
                gradH_num = grid.grad_H(num_sol)
                gradH_ex = grid.grad_H(exact_sol)
                error_grad = (gradH_num[0] - gradH_ex[0], gradH_num[1] - gradH_ex[1])
                grad_h_norm_p_sq_errors[var_name] = (
                    grid.norm_p(error_grad[0], error_grad[1]) ** 2
                )
            else:
                grad_h_norm_p_sq_errors[var_name] = 0.0  # Convenience

        return h_norm_sq_errors, grad_h_norm_p_sq_errors
        # --- End of: collect_errors

    # --- Collect data at t=0 ---

    h_norm_sq_errors_t0, grad_h_norm_p_sq_errors_t0 = collect_errors(
        initial_state_gfp, 0.0
    )
    time_series_data.append(
        TimeStepData(
            t=0.0,
            h_norm_sq_errors=h_norm_sq_errors_t0,
            grad_h_norm_p_sq_errors=grad_h_norm_p_sq_errors_t0,
        )
    )

    for step in range(num_steps):
        current_num_state = integrator.step(current_num_state, t0=current_t, dt=dt)
        current_t += dt

        # --- Collect data at current_t ---
        h_norm_sq_errors_tk, grad_h_norm_p_sq_errors_tk = collect_errors(
            current_num_state, current_t
        )
        time_series_data.append(
            TimeStepData(
                t=current_t,
                h_norm_sq_errors=h_norm_sq_errors_tk,
                grad_h_norm_p_sq_errors=grad_h_norm_p_sq_errors_tk,
            )
        )

        # --- Print progress ---
        if (step + 1) % steps_til_print == 0 or step == num_steps - 1:
            progress = (step + 1) / num_steps * 100
            elapsed = time.time() - t_start_sim
            # Reduce print frequency slightly
            if (step + 1) % (2 * steps_til_print) == 0 or step == num_steps - 1:
                print(
                    f"    Step {step + 1}/{num_steps} ({progress:.1f}%) completed. Elap.: {elapsed:.1f}s",
                    end="\r",
                )

    t_end_sim = time.time()
    print(f"\n    Simulation finished in {t_end_sim - t_start_sim:.2f} seconds.")
    assert np.isclose(
        current_t, Tf
    ), f"Final time mismatch: current_t={current_t}, Tf={Tf}"
    return time_series_data, dt


def calculate_combined_error_norm(
    time_series_data: list[TimeStepData], dt: float, integral_vars: List[str]
) -> float:
    """Calculates the combined max-integral error norm from time series data."""
    max_combined_norm_sq = 0.0
    current_integral_term = 0.0

    integrand_values = []
    for step_data in time_series_data:
        step_integrand = sum(
            step_data.grad_h_norm_p_sq_errors[var] for var in integral_vars
        )
        integrand_values.append(step_integrand)

    for k, step_data in enumerate(time_series_data):
        spatial_term_sq = sum(step_data.h_norm_sq_errors.values())
        if k > 0:
            current_integral_term += (
                0.5 * dt * (integrand_values[k - 1] + integrand_values[k])
            )

        combined_norm_sq_tk = spatial_term_sq + current_integral_term
        max_combined_norm_sq = max(max_combined_norm_sq, combined_norm_sq_tk)

    final_error_norm = np.sqrt(max_combined_norm_sq)
    print(f"    Combined Max-Integral Error Norm: {final_error_norm:.4e}")
    return final_error_norm


def _setup_simulation_instances(
    *,
    field_cls: p1.SemiDiscreteFieldBase,
    forcing_terms_cls: p1.ForcingTermsBase,
    mms_case_cls: p1.MMSCaseBase,
    integrator_cls: p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegratorBase,
    grid: p1.Grid,
    model: p1.DefaultModel01,
    variable_names: List[str],
    num_pc_steps: int,
    num_newton_steps: int,
) -> Tuple[p1.MMSCaseBase, p1.TimeIntegratorBase, p1.StateVars]:
    """
    Instantiates Exact Solution, Field, Integrator, and Initial GFP.
    Handles potential errors during instantiation.

    Returns:
        Tuple (exact_solution_instance, integrator_instance, initial_gfp_instance)
    Raises:
        TypeError, AttributeError, ValueError etc. if instantiation fails fundamentally.
    """
    # Instantiate Exact Solution
    # Assumes constructor: mms_case_cls(grid, model=model)
    mms_case = mms_case_cls(grid=grid, model=model)

    # Assumes constructor: forcing_terms_cls(*, mms_case, model)
    forcing_terms = forcing_terms_cls(mms_case=mms_case, model=model)

    # Instantiate Field
    # Assumes constructor: field_cls(*, grid, model, forcing_terms)
    current_field = field_cls(grid=grid, model=model, forcing_terms=forcing_terms)

    # Instantiate Integrator
    # Assumes constructor: integrator_cls(*, semi_discrete_field, num_pc_steps, num_newton_steps)
    integrator = integrator_cls(
        semi_discrete_field=current_field,
        num_pc_steps=num_pc_steps,
        num_newton_steps=num_newton_steps,
    )

    xx, yy = grid.xx, grid.yy

    # Instantiate Initial GFP State
    initial_state_dict = {v: getattr(mms_case, v)(0.0, xx, yy) for v in variable_names}
    initial_gfp = p1.StateVars(
        **initial_state_dict, model=model, hh=grid.hh, kk=grid.kk
    )

    return mms_case, integrator, initial_gfp


# --- Type Aliases for Clarity ---
# A full convergence report (FullCvgReport) is a dictionary with keys 'spatial' and 'temporal' (CvgType), effectively containing two inner convergence reports (one for spatial convergence, another for temporal convergence). Each such report (CvgReport) is a dictionary (i.e. record structure) with an 'errors' key and a list of errors. The key-value structre of CvgReport is for future extensibility, allowing for more complex reports later. The errors list can contain floats or None (for error cases).
# TODO: Formalize this with a dataclass or similar structure for enhanced functionality, better typing, etc.

CvgType = Literal["temporal", "spatial"]
CvgReportEntry = Literal["errors"]
NumericalList = List[float | None]  # Allow None for error
CvgReport = Dict[CvgReportEntry, NumericalList]
FullCvgReport = Dict[CvgType, CvgReport]

# TODO: Change to work with actual general factories, or stand-alone lambda constructors. Working with constructors directly is ad-hoc, and not much of a good idea for flexibility as it's inconvenient to ask for sameness of signature.
StudyConfig = Tuple[p1.SemiDiscreteFieldBase, p1.MMSCaseBase, p1.ForcingTermsBase, p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegratorBase, str]


# --- Main Study Function ---
def run_convergence_studies(
    study_configs: List[StudyConfig], study_params: Dict[str, Any]
) -> Dict[str, FullCvgReport]:
    """
    Runs spatial and temporal convergence studies using class types for configuration.

    Args:
        study_configs: List of tuples, each (Field_Class, ExactSolution_Class, label).
        study_params: Dict containing parameters like Tf, N_base_spatial, model instance etc.

    Returns:
        Dict mapping label to results (FullCvgReport).
    """

    all_results = {}

    # --- Extract study parameters ---
    # The idea is to use for default values. Can override if needed.
    # This is a bit of a hack, but it works for now.
    # TODO: Could use a dataclass for more structure.
    # NOTE: study_params should be a dictionary with all necessary keys, for now.
    # NOTE: Should study param labels be more explicitly checked? I believe using strings for labels is fine, but proper checking is currenctly missing.
    variable_names = study_params.get("variable_names", ["cp", "T", "cl", "cd", "cs"])
    integral_vars = study_params.get("integral_vars", ["T", "cl", "cd"])
    Tf = study_params["Tf"]
    model = study_params["model"]
    num_pc_steps = study_params.get("num_pc_steps", 1)
    num_newton_steps = study_params.get("num_newton_steps", 1)

    # --- Run convergence study cases ---
    for field_cls, mms_case_cls, forcing_terms_cls, integrator_cls, label in study_configs:
        print(f"\n===== Running Studies for Case: {label} =====")

        # Initialize results structure
        # This is a dictionary with keys 'spatial' and 'temporal', each containing.
        # TODO: Could use a dataclass for more structure, including the ability to do default constructors, copying, etc.
        case_results: FullCvgReport = {
            "spatial": {"errors": [], "rates": [], "statuses": []},
            "temporal": {"errors": [], "rates": [], "statuses": []},
        }

        # --- Spatial Study ---
        print("\n--- Starting Spatial Convergence Study ---")
        N_base = study_params["N_base_spatial"]
        num_refinements = study_params["num_spatial_refinements"]
        dt_fixed = study_params["dt_fixed_spatial"]
        spatial_errors: List[Optional[float]] = []

        refinement_factor = 2

        for k in range(num_refinements):
            t_start_level = time.time()
            N = N_base * (refinement_factor**k)
            M = N
            print(f"\n  Spatial Level {k} (N=M={N}, dt={dt_fixed:.1e})")
            grid = p1.make_uniform_grid(N, M)

            # Instantiate components for this grid level
            current_mms_case, integrator, initial_gfp = _setup_simulation_instances(
                field_cls=field_cls,
                mms_case_cls=mms_case_cls,
                forcing_terms_cls=forcing_terms_cls,
                integrator_cls=integrator_cls,
                grid=grid,
                model=model,
                variable_names=variable_names,
                num_pc_steps=num_pc_steps,
                num_newton_steps=num_newton_steps,
            )

            # Run simulation and calculate error
            time_series, adjusted_dt = run_simulation_collect_data(
                grid=grid,
                integrator=integrator,
                exact_sol_pack=current_mms_case,
                initial_state_gfp=initial_gfp,
                Tf=Tf,
                dt=dt_fixed,
                variable_names=variable_names,
                integral_vars=integral_vars,
            )
            # Note: using adjusted_dt for calc, but dt_fixed defines the level
            error_scalar = calculate_combined_error_norm(
                time_series, adjusted_dt, integral_vars
            )
            spatial_errors.append(error_scalar)

            t_end_level = time.time()
            print(
                f"  Spatial Level {k} finished in {t_end_level - t_start_level:.2f} seconds."
            )
        case_results["spatial"]["errors"] = spatial_errors
        computed_rates = calculate_observed_rates(spatial_errors, refinement_factor)
        case_results["spatial"]["rates"] = [err for (err, status) in computed_rates]
        case_results["spatial"]["statuses"] = [
            status for (err, status) in computed_rates
        ]

        # --- Temporal Study ---
        print("\n--- Starting Temporal Convergence Study ---")
        N_fixed = study_params["N_fixed_temporal"]
        M_fixed = N_fixed
        dt_base = study_params["dt_base_temporal"]
        num_refinements_temp = study_params["num_temporal_refinements"]
        temporal_errors: List[Optional[float]] = []

        # Setup fixed grid components once
        print(f"  Setting up fixed grid (N=M={N_fixed})...")
        grid_fixed = p1.make_uniform_grid(N_fixed, M_fixed)

        mms_case_fixed, integrator_prototype, initial_gfp_fixed = (
            _setup_simulation_instances(
                field_cls=field_cls,
                mms_case_cls=mms_case_cls,
                forcing_terms_cls=forcing_terms_cls,
                integrator_cls=integrator_cls,
                grid=grid_fixed,
                model=model,
                variable_names=variable_names,
                num_pc_steps=num_pc_steps,
                num_newton_steps=num_newton_steps,
            )
        )
        # We need the field instance for the integrator on each level
        field_fixed = integrator_prototype.semi_discrete_field

        integrator = integrator_cls(
            semi_discrete_field=field_fixed,
            num_pc_steps=num_pc_steps,
            num_newton_steps=num_newton_steps,
        )

        for k in range(num_refinements_temp):
            t_start_level = time.time()
            dt = dt_base / (refinement_factor**k)
            print(f"\n  Temporal Level {k} (dt={dt:.4e})")

            # Run simulation (pass fixed exact solution and initial state)
            time_series, adjusted_dt = run_simulation_collect_data(
                grid=grid_fixed,
                integrator=integrator,
                exact_sol_pack=mms_case_fixed,
                initial_state_gfp=initial_gfp_fixed,
                Tf=Tf,
                dt=dt,
                variable_names=variable_names,
                integral_vars=integral_vars,
            )
            error_scalar = calculate_combined_error_norm(
                time_series, adjusted_dt, integral_vars
            )
            temporal_errors.append(error_scalar)

            t_end_level = time.time()
            print(
                f"  Temporal Level {k} finished in {t_end_level - t_start_level:.2f} seconds."
            )
        case_results["temporal"]["errors"] = temporal_errors
        computed_rates = calculate_observed_rates(temporal_errors, refinement_factor)
        case_results["temporal"]["rates"] = [rate for (rate, status) in computed_rates]
        case_results["temporal"]["statuses"] = [
            status for (rate, status) in computed_rates
        ]

        all_results[label] = case_results
        print(f"\n===== Finished Studies for Case: {label} =====")

    return all_results

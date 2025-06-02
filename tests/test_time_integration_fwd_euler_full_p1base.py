# test_time_integration_fwd_euler_full_p1base.py

import pytest
import numpy as np
from numpy.testing import assert_allclose

import prob1base as p1
import prob1_mms_cases as p1mc
from utils_for_testing import observed_rates_report
import mms_trial_utils as mtu

R0_test_fwd_euler: float = 8.3144621
Ea_test_fwd_euler: float = 1.60217662e-19


@pytest.fixture(
    scope="session"
)  # Model consts can be session-scoped if unchanged across tests in this file
def p1_default_model_consts_fwd_euler() -> p1.ModelConsts:
    """Provides prob1base.ModelConsts with values from the original test's default."""
    return p1.ModelConsts(
        R0=R0_test_fwd_euler,
        Ea=Ea_test_fwd_euler,
        K1=1e-2,
        K2=1e-2,
        K3=1e-2,
        K4=1e-2,
        DT=1e-3,
        Dl_max=8.01e-4,
        phi_l=1e-5,
        gamma_T=1e-9,
        Kd=1e-8,
        Sd=10,
        Dd_max=2.46e-6,
        phi_d=1e-5,
        phi_T=Ea_test_fwd_euler / R0_test_fwd_euler,
        r_sp=5e-2,
        T_ref=300,
    )


# --- Test Parameters ---
GRID_SIZES_FE = [5, 15, 35]
grid_size_parameterization_fe = pytest.mark.parametrize("grid_n", GRID_SIZES_FE)

# Reduced T_TEST_VALUES for faster execution, can be expanded
POWERS_OF_100_FE = np.power(100.0, np.arange(-1, 1))  # 1/100, 1, 100
T_TEST_VALUES_FE = np.sort(np.concatenate([POWERS_OF_100_FE, 5 * POWERS_OF_100_FE]))
t_test_parameterization_fe = pytest.mark.parametrize("t_test", T_TEST_VALUES_FE)

RTOL_FE_CONSISTENCY_P1 = 1e-3 # Atol is based on the dt-dependent tolerance. Should this be as well?
ORDER_ASSERT_TOLERANCE_P1 = 0.15  # A lot?


@pytest.fixture(scope="function")
def fe_runner_steup(grid_n: int, p1_default_model_consts_fwd_euler: p1.ModelConsts):
    """
    Sets up Grid, Model, MMSCase, ForcingTerms, SemiDiscreteField, and ForwardEulerIntegrator.
    """
    grid = p1.make_uniform_grid(N=grid_n, M=grid_n)
    model = p1.DefaultModel02(mc=p1_default_model_consts_fwd_euler)
    mms_trial = mtu.MMSTrial(
        grid=grid,
        model=model,
        mms_case_cls=p1mc.MMSCaseExpSin,
        field_cls=p1.SemiDiscreteField_CsTriple,
        forcing_terms_cls=p1.ForcingTerms_CsTriple,
        integrator_cls=p1.ForwardEulerIntegrator,
    )

    print(f"\n  Setting up Forward Euler test for N=M={grid_n}...")
    return grid, model, mms_trial


@grid_size_parameterization_fe
@t_test_parameterization_fe
def test_fwd_euler_full_one_step_consistency_p1base(
    fe_runner_steup, t_test: float, grid_n: int
):
    """
    Tests consistency of one Forward Euler step against the exact MMS solution.
    """
    grid, model, mms_trial = fe_runner_steup

    # Estimate stable dt
    # Using Dl_max and Dd_max from the model, and DT.
    mc = model
    max_D_coeff = max(mc.DT, mc.Dl_max, mc.Dd_max, 1e-9)  # Ensure positive
    assert max_D_coeff > 0, "Maximum diffusion coefficient must be positive."
    h_sq = (1.0 / grid_n) ** 2
    dt_stable_est = h_sq / (4.0 * max_D_coeff)  # using CFL = 0.25

    assert np.isfinite(
        dt_stable_est
    ), f"Estimated stable dt is not finite: {dt_stable_est:.2e} for grid size N={grid_n}."

    dt_test = min(1e-2, dt_stable_est * 0.5)  # Attempts to get a small valid stable dt.

    print(
        f"  Running Consistency Test: N={grid_n}, t={t_test:.2e}, dt={dt_test:.1e} (Est. stable dt ~ {dt_stable_est:.1e})"
    )

    t_start = t_test
    t_end = t_start + dt_test

    # MMS Trial run for 1 step only.
    error_summary = mms_trial.run_for_errors(Tf=t_end, dt=dt_test, t0=t_start)

    # NOTE: Interesting ATOL determination based on known consistency order. Helps in, more systematically, identify reasonable tollerances.
    # For first-order methods, we can use a constant times Δt as the absolute tolerance. This constant might need to be large (adjusting it is trial-and-error, to my knowledge, for now). Once determined, for a single test case (e.g. single choice of t_test and grid_n), it then can be tried/used for all variables. The idea is to determine this leading constant based on one test case, and then use it for all other tests with the same grid_n and t_test (helps in avoiding the "choosing leading constant to make it work" bias/problem).
    dt_dependent_atol_cte = 100 # Magical constant, determined through trial and error.
    dt_dependent_atol = dt_dependent_atol_cte * dt_test # must be a constant times Δt, as fwdeuler is first-order accurate in time.

    # Compare interior points
    for var_name in ["cp", "T", "cl", "cd", "cs"]:
        this_var_error = error_summary.per_variable_sup_errors[var_name]
        print(f"    Error for {var_name}: {this_var_error:.5e}")
        assert this_var_error == pytest.approx(
            0.0, abs=dt_dependent_atol, rel=RTOL_FE_CONSISTENCY_P1
        ), f"{var_name} consistency failed N={grid_n}, t={t_test}, dt={dt_test}"


@grid_size_parameterization_fe
@t_test_parameterization_fe  # Using the same start times for temporal order test
def test_fwd_euler_full_temporal_order_p1base(
    fe_runner_steup, t_test: float, grid_n: int
):
    """
    Verifies the first-order temporal accuracy (p=1) of the ForwardEulerIntegrator.
    Calculates observed order for EACH variable individually using grid.norm_H.
    """
    _grid, model, mms_trial = fe_runner_steup
    mc = model

    # Estimate stable dt for the chosen grid_n
    mc = model
    max_D_coeff = max(mc.DT, mc.Dl_max, mc.Dd_max, 1e-9)  # Ensure positive
    assert max_D_coeff > 0, "Maximum diffusion coefficient must be positive."
    h_sq = (1.0 / grid_n) ** 2
    dt_stable_est = h_sq / (4.0 * max_D_coeff)  # using CFL = 0.25

    assert np.isfinite(
        dt_stable_est
    ), f"Estimated stable dt is not finite: {dt_stable_est:.2e} for grid size N={grid_n}."

    base_dt = min(4e-4, dt_stable_est * 0.5)  # Start with a reasonably small, stable dt

    print(
        f"  Running Temporal Order Test: N={grid_n}, t_start={t_test:.2e}, base_dt={base_dt:.1e} (Est. stable dt ~ {dt_stable_est:.1e})"
    )

    num_refinements = 4
    refinement_ratio = 2.0

    actual_dts = [base_dt / (refinement_ratio**k) for k in range(num_refinements)]

    errors_all_vars = {"cp": [], "T": [], "cl": [], "cd": [], "cs": []}
    errors_combined = []  # Store sum of squared H-norms for combined order

    t_start = t_test

    for dt_k in actual_dts:
        print(f"    Testing dt = {dt_k:.2e}...")
        t_end = t_start + dt_k

        error_summary = mms_trial.run_for_errors(Tf=t_end, dt=dt_k, t0=t_start)

        print_err_line = []
        for var_name in errors_all_vars.keys():
            err_norm = error_summary.per_variable_sup_errors[var_name]
            errors_all_vars[var_name].append(err_norm)
            print_err_line.append(f"{var_name}={err_norm:.2e}")   

        errors_combined.append(error_summary.overall_combined_error)
        print(
            f"      Combined Error (sqrt sum sq) = {errors_combined[-1]:.3e} ({', '.join(print_err_line)})"
        )

    # --- Calculate and Print Observed Order for Each Variable ---
    print(f"  Observed Orders (finest steps, N={grid_n}, t_start={t_test:.2e}):")
    expected_order_fe = 1.0

    for var_name, error_list in errors_all_vars.items():
        print(f"    Observed Orders for {var_name}:")
        observed_rates_report(
            error_list,
            expected_rate=expected_order_fe,
            tolerance=ORDER_ASSERT_TOLERANCE_P1,
        )

    # Also check combined error order
    errors_combined_norm = [np.sqrt(err_sq) for err_sq in errors_combined]
    print(f"    Observed Orders for Combined Error Norm:")
    observed_rates_report(
        errors_combined_norm,
        expected_rate=expected_order_fe,
        tolerance=ORDER_ASSERT_TOLERANCE_P1,
    )
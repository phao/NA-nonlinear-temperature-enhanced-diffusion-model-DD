import pytest
import numpy as np
import sympy
import prob1base as p1
from cvg_studies_base import calculate_observed_rates, RateStatus
from typing import List, Optional, Dict, Tuple, Any
import math  # Ensure math is imported

# --- Common Symbolic Setup ---
from prob1base import t_sym, x_sym, y_sym

pi_sym = sympy.pi

# --- MMS for Coupled T-cl Test ---
# Re-use the same exact solutions as the failing implicit test
T_exact_sym_tc = (
    10 * sympy.sin(pi_sym * x_sym) * sympy.sin(pi_sym * y_sym) * (1 + t_sym)
)
cl_exact_sym_tc = (
    (sympy.cos(pi_sym * x_sym) ** 2 - 1) * (1 - sympy.cos(pi_sym * y_sym) ** 2) * t_sym
)
# Others are zero
cp_sym_zero_tc = sympy.S(0)
cd_sym_zero_tc = sympy.S(0)
cs_sym_zero_tc = sympy.S(0)

# --- Test Model for Coupled T-cl ---
# Same model as the implicit test, activating relevant constants
test_DT = 1e-3
test_Dl_max = 8.0e-4
tc_feuler_model_consts = p1.ModelConsts(
    R0=p1.R0,
    Ea=p1.Ea,
    K1=0.0,
    K2=0.0,
    K3=1e-3,
    K4=5e-3,
    DT=test_DT,
    Dl_max=test_Dl_max,
    phi_l=0.0,
    gamma_T=1e-9,
    Kd=0.0,
    Sd=0.0,
    Dd_max=0.0,
    phi_d=0.0,
    phi_T=0.0,
    r_sp=0.0,
)
tc_feuler_model = p1.DefaultModel01(mc=tc_feuler_model_consts)

# --- Test Parameters ---
N_list_tc_feuler = [8, 16, 32, 64]
Tf_tc_feuler = 0.001  # Use a very short Tf for FE stability
CFL_target = 0.1  # Target CFL for dt calculation


def test_T_cl_coupled_feuler_spatial_convergence():
    """
    Verifies spatial convergence for the coupled T-cl system using the
    Forward Euler integrator and an MMS setup via MMSCaseSymbolic.
    """
    final_errors_T_H_norm = []
    final_errors_cl_H_norm = []
    D_max = max(tc_feuler_model.DT, tc_feuler_model.Dl(0.0))

    print("\n--- Running Coupled T-cl FEuler Spatial Conv Test ---")
    for N in N_list_tc_feuler:
        print(f"  Testing N={N}...")
        grid = p1.make_uniform_grid(N, N)
        variable_names_tc = ["cp", "T", "cl", "cd", "cs"]

        # --- Calculate dt for this level: dt =  CFL * h^2 / D_max ---
        h = 1.0 / N  # dx = dy = 1/N
        dt_level = CFL_target * (h**2) / D_max

        print(f"    Using dt = {dt_level:.3e} (based on h^2 scaling)")

        # --- Instantiate MMS Spec ---
        try:
            mms_case = p1.MMSCaseSymbolic(
                grid=grid,
                model=tc_feuler_model,
                cp_sym_expr=cp_sym_zero_tc,
                T_sym_expr=T_exact_sym_tc,
                cl_sym_expr=cl_exact_sym_tc,
                cd_sym_expr=cd_sym_zero_tc,
                cs_sym_expr=cs_sym_zero_tc,
            )
        except Exception as e:
            pytest.fail(f"Failed to instantiate MMSCaseSymbolic for N={N}: {e}")

        cs2_forcing = p1.ForcingTerms_CsTriple(mms_case=mms_case, model=tc_feuler_model)

        # --- Instantiate Field and FORWARD EULER Integrator ---
        field = p1.SemiDiscreteField_CsTriple(
            grid=grid, model=tc_feuler_model, forcing_terms=cs2_forcing
        )
        integrator = p1.ForwardEulerIntegrator(semi_discrete_field=field)

        # --- Instantiate Initial State ---
        xx, yy = grid.xx, grid.yy
        initial_state = p1.state_from_mms_when(mms_case=mms_case, t=0.0, grid=grid)

        # --- Run Simulation ---
        current_state_sim = initial_state
        current_t_sim = 0.0
        # Ensure num_steps calculation uses the dt specific to this level
        num_steps_sim = math.ceil(Tf_tc_feuler / dt_level)
        actual_dt = Tf_tc_feuler / num_steps_sim  # Refine dt to hit Tf exactly

        if abs(actual_dt - dt_level) > 1e-6 * dt_level:
            print(f"    Adjusted dt slightly to: {actual_dt:.3e}")

        for _step in range(num_steps_sim):
            current_state_sim = integrator.step(
                current_state_sim, t0=current_t_sim, dt=actual_dt
            )
            current_t_sim += actual_dt

        # --- Calculate Final Errors for T and cl ---
        final_T_numerical = current_state_sim.T
        final_cl_numerical = current_state_sim.cl
        final_T_exact = mms_case.T(Tf_tc_feuler, xx, yy)
        final_cl_exact = mms_case.cl(Tf_tc_feuler, xx, yy)

        error_T_final = final_T_numerical - final_T_exact
        error_cl_final = final_cl_numerical - final_cl_exact

        error_norm_T = grid.norm_H(error_T_final) + np.finfo(float).eps
        error_norm_cl = grid.norm_H(error_cl_final) + np.finfo(float).eps

        print(f"    N={N}, Final H-Norm Error(T) = {error_norm_T:.4e}")
        print(f"    N={N}, Final H-Norm Error(cl) = {error_norm_cl:.4e}")
        final_errors_T_H_norm.append(error_norm_T)
        final_errors_cl_H_norm.append(error_norm_cl)

    # --- Calculate and Assert observed rates for T ---
    print("\n  Calculating observed rates for T...")
    valid_errors_T = [
        e for e in final_errors_T_H_norm if e is not None and np.isfinite(e) and e >= 0
    ]
    if len(valid_errors_T) < 3:
        pytest.fail(
            f"Insufficient valid T error points ({len(valid_errors_T)}) for rate calculation."
        )
    rates_with_status_T = calculate_observed_rates(
        valid_errors_T, refinement_factor=2.0
    )
    assert rates_with_status_T, "Rate calculation for T failed."
    final_rate_T, final_status_T = rates_with_status_T[-1]
    print(f"  Rates(T): {rates_with_status_T}")
    print(f"  Final Rate(T): {final_rate_T:.3f}, Status: {final_status_T}")
    assert (
        final_status_T == RateStatus.OK
    ), f"Rate calculation for T status: {final_status_T}"
    target_order = 2.0
    # Allow more tolerance as FE is only O(dt) accurate, need small Tf/dt to see spatial O(h^2)
    assert final_rate_T == pytest.approx(
        target_order, abs=0.2
    ), f"Expected spatial order {target_order} for T, but got {final_rate_T:.3f}"

    # --- Calculate and Assert observed rates for cl ---
    print("\n  Calculating observed rates for cl...")
    valid_errors_cl = [
        e for e in final_errors_cl_H_norm if e is not None and np.isfinite(e) and e >= 0
    ]
    if len(valid_errors_cl) < 3:
        pytest.fail(
            f"Insufficient valid cl error points ({len(valid_errors_cl)}) for rate calculation."
        )
    rates_with_status_cl = calculate_observed_rates(
        valid_errors_cl, refinement_factor=2.0
    )
    assert rates_with_status_cl, "Rate calculation for cl failed."
    final_rate_cl, final_status_cl = rates_with_status_cl[-1]
    print(f"  Rates(cl): {rates_with_status_cl}")
    print(f"  Final Rate(cl): {final_rate_cl:.3f}, Status: {final_status_cl}")
    assert (
        final_status_cl == RateStatus.OK
    ), f"Rate calculation for cl status: {final_status_cl}"
    # Allow more tolerance again
    assert final_rate_cl == pytest.approx(
        target_order, abs=0.2
    ), f"Expected spatial order {target_order} for cl, but got {final_rate_cl:.3f}"

    print("\n--- Coupled T-cl FEuler Spatial Convergence Test Passed ---")

import math
from typing import List, Optional, Dict, Tuple, Any

import pytest
import numpy as np
import sympy
import prob1base as p1

# Assuming calculate_observed_rates is accessible
from cvg_studies_base import calculate_observed_rates, RateStatus

# --- Common Symbolic Setup ---
t_sym, x_sym, y_sym = sympy.symbols("t x y")
pi_sym = sympy.pi

# --- MMS Solution for T Gradient Test ---
# T = sin(pi*x)sin(pi*y) * exp(-t)
T_exact_sym = sympy.sin(pi_sym * x_sym) * sympy.sin(pi_sym * y_sym) * sympy.exp(-t_sym)
cp_sym_zero = sympy.S(0)
cl_sym_zero = sympy.S(0)
cd_sym_zero = sympy.S(0)
cs_sym_zero = sympy.S(0)

# --- Define Simplified Model (only DT matters here) ---
test_model_consts_t = p1.ModelConsts(
    R0=p1.R0,
    Ea=p1.Ea,
    K1=0.0,
    K2=0.0,
    K3=0.0,
    K4=0.0,
    DT=1e-3,  # Non-zero diffusion for T
    Dl_max=0.0,
    phi_l=0.0,
    gamma_T=0.0,
    Kd=0.0,
    Sd=0.0,
    Dd_max=0.0,
    phi_d=0.0,
    phi_T=0.0,
    r_sp=0.0,
)
test_model_t = p1.DefaultModel02(mc=test_model_consts_t)

# --- Analytical Gradient Calculation (needed for final error check) ---
dx_T_exact_sym = sympy.diff(T_exact_sym, x_sym)
dy_T_exact_sym = sympy.diff(T_exact_sym, y_sym)

# --- Test Parameters ---
N_list_grad_step = [8, 16, 32, 64]
# Let dt = C * h^2 = C / N^2 to ensure spatial error dominates
# Choose C such that dt is reasonably small but not excessively so.
# If C=0.1: N=8->dt=0.1/64=0.00156; N=64->dt=0.1/4096=2.4e-5
DT_SCALING_CONST = 0.1  # Adjust if needed


def test_gradient_T_spatial_convergence_one_step():
    """
    Verifies the spatial convergence order of the gradient's error after
    one time step. Uses dt ~ h^2 scaling.
    Compares numerical gradient of numerical solution vs
            analytical gradient of exact solution at t=dt.
    Uses the norm_p for measuring the gradient error.
    """
    gradient_errors_one_step_p_norm = []
    variable_names_t = ["cp", "T", "cl", "cd", "cs"]  # Only T matters

    print("\n--- Running Gradient(T) Spatial Convergence Test (One Step dt~h^2) ---")
    for N in N_list_grad_step:
        h_approx = 1.0 / N
        dt = DT_SCALING_CONST * (h_approx**2)
        t1 = dt
        print(f"  Testing N={N} (dt={dt:.3e})...")
        grid = p1.make_uniform_grid(N, N)

        # --- Instantiate MMS Spec using MMSCaseSymbolic ---
        # This calculates exact solutions and FORCING TERMS
        mms_case = p1.MMSCaseSymbolic(
            grid=grid,
            model=test_model_t,
            cp_sym_expr=cp_sym_zero,
            T_sym_expr=T_exact_sym,
            cl_sym_expr=cl_sym_zero,
            cd_sym_expr=cd_sym_zero,
            cs_sym_expr=cs_sym_zero,
        )

        # --- Use the FORCING terms derived numerically by MMSCaseSymbolic ---
        # We need fT so the integrator solves the correct modified PDE
        mms_forcing_terms = p1.ForcingTerms_CsTriple(model=test_model_t, mms_case=mms_case)
        forcing_terms_dict = {
            "fcp": lambda t, xx, yy: grid.make_full0(),
            "fT": mms_forcing_terms.fT,  # Use the forcing term for T
            "fcl": lambda t, xx, yy: grid.make_full0(),
            "fcd": lambda t, xx, yy: grid.make_full0(),
            "fcs": lambda t, xx, yy: grid.make_full0(),
        }
        forcing_terms = p1.ForcingTermsFromDict(forcing_terms_dict)

        # --- Instantiate Field and Integrator ---
        field = p1.SemiDiscreteField_CsTriple(
            grid=grid, model=test_model_t, forcing_terms=forcing_terms
        )
        # Use the main integrator
        integrator = p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_CsTriple(
            semi_discrete_field=field,
            num_pc_steps=1,  # Use parameters consistent with cvg_studies?
            num_newton_steps=1,
        )

        # --- Initial State (t=0) ---
        xx, yy = grid.xx, grid.yy
        initial_state_dict = {v: getattr(mms_case, v)(0.0, xx, yy) for v in variable_names_t}
        gfp_t0 = p1.StateVars(
            **initial_state_dict, model=test_model_t, hh=grid.hh, kk=grid.kk
        )

        # --- Take One Time Step ---
        gfp_t1_numerical = integrator.step(gfp_t0, t0=0.0, dt=dt)

        # --- Get Numerical Gradient at t1 ---
        gradH_T_t1_numerical = grid.grad_H(gfp_t1_numerical.T)

        # --- Get Analytical Gradient of Exact Solution at t1 ---
        T_exact_t1 = mms_case.T(t1, xx, yy)  # Exact T at t1
        gradH_T_t1_exact = grid.grad_H(T_exact_t1)
        
        # --- Calculate Error in the Gradient at t1 ---
        error_grad_t1 = (
            gradH_T_t1_numerical[0] - gradH_T_t1_exact[0],
            gradH_T_t1_numerical[1] - gradH_T_t1_exact[1],
        )

        # --- Compute p-norm of the gradient error ---
        error_norm = (
            grid.norm_p(error_grad_t1[0], error_grad_t1[1]) + np.finfo(float).eps
        )
        print(f"    N={N}, p-Norm(Gradient Error @ t=dt) = {error_norm:.4e}")
        gradient_errors_one_step_p_norm.append(error_norm)

    # --- Calculate observed rates ---
    print("\n  Calculating observed rates for one-step gradient error...")
    valid_errors = [
        e
        for e in gradient_errors_one_step_p_norm
        if e is not None and np.isfinite(e) and e >= 0
    ]
    if len(valid_errors) < 3:
        pytest.fail(
            f"Insufficient valid gradient error points ({len(valid_errors)}) for rate calculation."
        )

    # Refinement factor is 2.0 for N
    rates_with_status = calculate_observed_rates(valid_errors, refinement_factor=2.0)

    # --- Assert convergence order ---
    print(f"  Rates with status: {rates_with_status}")
    assert rates_with_status, "Rate calculation failed for one-step gradient error."

    final_rate, final_status = rates_with_status[-1]

    print(
        f"  Final observed rate for one-step gradient error: {final_rate:.3f}, Status: {final_status}"
    )
    assert (
        final_status == RateStatus.OK
    ), f"Rate calculation did not return OK status: {final_status}"

    # NOW we expect 2nd order spatial convergence for this error
    target_order = 2.0
    assert final_rate >= target_order - 0.1, f"Expected spatial order {target_order} for one-step gradient error, but got {final_rate:.3f}"

    print("--- One-Step Gradient(T) Spatial Convergence Test Passed ---")

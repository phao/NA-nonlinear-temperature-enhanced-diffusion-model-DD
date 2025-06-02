import math
from typing import List, Literal, Optional, Dict, Tuple, Any

import pytest
import numpy as np
import sympy
import prob1base as p1

from cvg_studies_base import (
    calculate_observed_rates,
    RateStatus,
)

import mms_trial_utils as mms_utils


# --- Common Symbolic Setup ---
from prob1base import x_sym, y_sym, t_sym
pi_sym = sympy.pi

@pytest.fixture()
def spatial_refinements_N_list() -> List[int]:
    """
    Provides a list of spatial refinements N for convergence studies.
    This is used in tests to ensure consistent grid sizes across different tests.
    """
    base_N = 16
    n_refinements = 4  # Number of refinements
    return [base_N * (2 ** i) for i in range(n_refinements)]

@pytest.fixture()
def cvg_test_Tf() -> float:
    return 1e-4 # Small final time for unit/integration tests. Actual convergence studies are done with larger times elsewhere. 

@pytest.fixture()
def h_to_dt_const() -> float:
    """
    In spatial convergence studyies, we use dt = cte * h² for each considered h. This fixture specifies this `cte`.
    """
    return 1e-2

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

    match cmp_type:
        case "equal":
            assert_if_on(
                final_rate == pytest.approx(target_order, abs=order_abs_tol),
                f"Expected spatial order {target_order} for {name}, but got {final_rate}",
            )
        case "least":
            assert_if_on(
                final_rate >= target_order - order_abs_tol,
                f"Expected spatial order at least {target_order} for {name}, but got {final_rate:.3f}",
            )


def test_FT_laplacian_spatial_convergence_symbolic():
    """
    Verifies the spatial convergence order of the FT component's Laplacian term
    using the MMSCaseSymbolic class for MMS setup. It compares the
    numerical operator applied to T_exact(t=0) against the analytical value
    DT*lap(T_exact(t=0)).
    """

    # --- MMS Solution for FT Test ---
    # Use the same simple solution as before
    T_exact_sym_ft = sympy.sin(pi_sym * x_sym) * sympy.sin(pi_sym * y_sym) * (1 + t_sym)
    # Other fields can be zero for this isolated test
    cp_sym_zero = sympy.S(0)
    cl_sym_zero = sympy.S(0)
    cd_sym_zero = sympy.S(0)
    cs_sym_zero = sympy.S(0)

    # --- Analytical Operator for Comparison ---
    # FT operator's spatial part is DT*lap(T) (since K3=0 in test model)
    lap_T_analytic_sym_ft = sympy.diff(T_exact_sym_ft, x_sym, 2) + sympy.diff(
        T_exact_sym_ft, y_sym, 2
    )
    # Need to lambdify this *multiplied by DT* later, once model DT is known

    # Define the simplified model just for this FT test
    ft_test_model_consts = p1.ModelConsts(
        R0=p1.R0,
        Ea=p1.Ea,
        K1=0.0,
        K2=0.0,
        K3=0.0,
        K4=0.0,  # K3 = 0 is crucial
        DT=1e-3,  # Use a non-zero DT
        Dl_max=0.0,
        phi_l=0.0,
        gamma_T=0.0,
        Kd=0.0,
        Sd=0.0,
        Dd_max=0.0,
        phi_d=0.0,
        phi_T=0.0,
        r_sp=0.0,
        T_ref=300.0
    )
    ft_test_model = p1.DefaultModel02(mc=ft_test_model_consts)
    # Lambdify the analytical spatial operator FT part
    DT_val = ft_test_model.DT  # Get the numerical value
    L_spatial_analytic_sym_ft = DT_val * lap_T_analytic_sym_ft
    L_spatial_analytic_func_ft = sympy.lambdify(
        (t_sym, x_sym, y_sym), L_spatial_analytic_sym_ft, "numpy"
    )

    # --- Test Parameters ---
    N_list_ft = [8, 16, 32, 64]
    test_time_ft = 0.0  # Evaluate spatial operator at t=0

    truncation_errors_H_norm = []

    print("\n--- Running FT Spatial Conv Test (using MMSCaseSymbolic) ---")
    for N in N_list_ft:
        print(f"  Testing N={N}...")
        grid = p1.make_uniform_grid(N, N)
        xx, yy = grid.xx, grid.yy

        # --- Instantiate MMS Spec using MMSCaseSymbolic ---
        try:
            mms_spec = p1.MMSCaseSymbolic(
                grid=grid,
                model=ft_test_model,
                cp_sym_expr=cp_sym_zero,
                T_sym_expr=T_exact_sym_ft,
                cl_sym_expr=cl_sym_zero,
                cd_sym_expr=cd_sym_zero,
                cs_sym_expr=cs_sym_zero,
                t_var=t_sym,
                x_var=x_sym,
                y_var=y_sym,  # Use default symbols
            )
        except Exception as e:
            pytest.fail(f"Failed to instantiate MMSCaseSymbolic for N={N}: {e}")

        # Define forcing terms dictionary - uses methods from mms_spec
        # Crucially, MMSCaseSymbolic inherits fT which calls numerical T, cp etc.
        # For this test, fT *should* evaluate numerically close to L_spatial_analytic_ft,
        # but for the truncation error test we want operator(u_exact) vs analytical_operator(u_exact).
        # Let's redefine fT locally to be zero.
        # Instantiate Field
        field = p1.SemiDiscreteField_CsTriple(
            grid=grid, model=ft_test_model, forcing_terms=p1.NoForcingTerms(grid=grid)
        )

        # --- Evaluate numerical operator FT on exact state ---
        # 1. Get exact state StateVars at test_time
        gfp_exact_state = p1.StateVars(
            cp=mms_spec.cp(test_time_ft, xx, yy),
            T=mms_spec.T(test_time_ft, xx, yy),
            cl=mms_spec.cl(test_time_ft, xx, yy),
            cd=mms_spec.cd(test_time_ft, xx, yy),
            cs=mms_spec.cs(test_time_ft, xx, yy),
            model=ft_test_model,
            hh=grid.hh,
            kk=grid.kk,
        )

        # 2. Apply numerical FT operator
        FT_numerical_grid = field.FT(gfp_exact_state, test_time_ft)

        # --- Get Analytical value of the spatial operator DT*lap(T) ---
        L_spatial_analytic_grid = L_spatial_analytic_func_ft(
            test_time_ft, grid.xx, grid.yy
        )

        # --- Calculate Truncation Error ---
        # TE = Operator_Numerical(u_exact) - Operator_Analytical(u_exact)
        truncation_error_grid = FT_numerical_grid - L_spatial_analytic_grid

        # --- Compute H-norm of truncation error ---
        error_norm = grid.norm_H(truncation_error_grid) + np.finfo(float).eps
        print(f"    N={N}, H-Norm(Truncation Error) = {error_norm:.4e}")
        truncation_errors_H_norm.append(error_norm)

    deliverable_from_test_caculate_observed_rates(
        name="H-Norm(FT Truncation)",
        errors=truncation_errors_H_norm,
        with_asserts=True
    )

    print("--- FT Symbolic Test Passed ---")


# --- test_Fcl_spatial_convergence_symbolic function definition ---
def test_Fcl_spatial_convergence_symbolic():
    """
    Verifies the spatial convergence order of the Fcl component using
    MMSCaseSymbolic. Compares the numerical operator Fcl applied to the exact
    state against the analytical spatial RHS derived from the PDE definition.
    """

    # --- Setup for Fcl Test (Corrected Symbolic Calculation) ---

    # MMS Solutions needed for Fcl
    cl_exact_sym_fcl = (
        sympy.sin(pi_sym * x_sym) * sympy.sin(pi_sym * y_sym) * (1 + t_sym**2)
    )
    T_exact_sym_fcl = (
        sympy.cos(0.5 * pi_sym * x_sym) * sympy.cos(0.5 * pi_sym * y_sym) * (1 + t_sym)
    )
    cp_exact_sym_fcl = (
        sympy.sin(pi_sym * x_sym) ** 2 * sympy.sin(pi_sym * y_sym) ** 2 * (1 + 0.5 * t_sym)
    )
    # Zero placeholders
    cd_sym_zero_fcl = sympy.S(0)
    cs_sym_zero_fcl = sympy.S(0)

    # Fcl Test Model Instance (numerical methods now)
    fcl_test_model_consts = p1.ModelConsts(
        R0=p1.R0,
        Ea=p1.Ea,
        K1=0.0,
        K2=0.0,
        K3=0.0,
        K4=1e-2,
        DT=0.0,
        Dl_max=8.0e-4,
        phi_l=1e-5,
        gamma_T=1e-9,
        Kd=0.0,
        Sd=0.0,
        Dd_max=0.0,
        phi_d=0.0,
        phi_T=0.0,
        r_sp=0.0,
    )
    fcl_test_model = p1.DefaultModel02(mc=fcl_test_model_consts)

    # --- CORRECTED Analytical Operator Calculation for Fcl ---
    # Perform ALL differentiation using symbols, then substitute MMS expr

    # 1. Define base abstract symbols if needed, or use MMS expressions carefully
    cp_abstract_sym = sympy.Symbol("cp_abstract_sym")  # Temp symbol for differentiation
    T_abstract_sym = sympy.Symbol("T_abstract_sym")  # Temp symbol for differentiation

    # 2. Define model coefficient expressions using these abstract symbols
    Dl_abstract_expr = fcl_test_model.Dl_max * sympy.exp(
        -fcl_test_model.phi_l * cp_abstract_sym
    )
    V1_abstract_expr = fcl_test_model.gamma_T * T_abstract_sym

    # 3. Calculate required derivatives w.r.t. abstract symbols
    dDl_dcp_abstract_expr = sympy.diff(Dl_abstract_expr, cp_abstract_sym)
    dV1_dT_abstract_expr = sympy.diff(V1_abstract_expr, T_abstract_sym)

    # 4. Substitute the MMS expressions into these derivative forms *and* the base forms
    #    Substitution mapping:
    sub_map = {cp_abstract_sym: cp_exact_sym_fcl, T_abstract_sym: T_exact_sym_fcl}
    Dl_on_MMS = Dl_abstract_expr.subs(sub_map)
    dDl_dcp_on_MMS = dDl_dcp_abstract_expr.subs(sub_map)
    V1_on_MMS = V1_abstract_expr.subs(sub_map)
    dV1_dT_on_MMS = (
        dV1_dT_abstract_expr  # This one is constant, no substitution needed, or subs({})
    )

    # 5. Calculate needed derivatives of MMS expressions (these were correct before)
    dx_cl_sym = sympy.diff(cl_exact_sym_fcl, x_sym)
    dy_cl_sym = sympy.diff(cl_exact_sym_fcl, y_sym)
    lap_cl_sym = sympy.diff(dx_cl_sym, x_sym) + sympy.diff(dy_cl_sym, y_sym)
    dx_cp_sym = sympy.diff(cp_exact_sym_fcl, x_sym)
    dy_cp_sym = sympy.diff(cp_exact_sym_fcl, y_sym)
    dx_T_sym = sympy.diff(T_exact_sym_fcl, x_sym)
    dy_T_sym = sympy.diff(T_exact_sym_fcl, y_sym)

    # 6. Assemble the symbolic spatial operator using substituted expressions
    analytical_rhs_sym_fcl = (
        dDl_dcp_on_MMS * (dx_cp_sym * dx_cl_sym + dy_cp_sym * dy_cl_sym)
        + Dl_on_MMS * lap_cl_sym
        - V1_on_MMS * dx_cl_sym
        # - V2_sym * dy_cl_sym # Still zero
        - (cl_exact_sym_fcl + 1) * (dV1_dT_on_MMS * dx_T_sym)  # + dV2*...=0
        - fcl_test_model.K4 * cp_exact_sym_fcl * (cl_exact_sym_fcl + 1)
    )

    # 7. Lambdify the final analytical expression
    L_spatial_analytic_func_fcl = sympy.lambdify(
        (t_sym, x_sym, y_sym), analytical_rhs_sym_fcl, "numpy"
    )

    # --- Fcl Test Parameters (keep as before) ---
    N_list_fcl_sym = [8, 16, 32, 64]
    test_time_fcl = 0.1

    truncation_errors_H_norm = []

    print("\n--- Running Fcl Spatial Conv Test (using MMSCaseSymbolic) ---")
    for N in N_list_fcl_sym:
        print(f"  Testing N={N}...")
        grid = p1.make_uniform_grid(N, N)
        xx, yy = grid.xx, grid.yy

        # --- Instantiate MMS Spec using MMSCaseSymbolic ---
        mms_spec = p1.MMSCaseSymbolic(
            grid=grid,
            model=fcl_test_model,  # Use the model with only K4/Dl/V1 active
            cp_sym_expr=cp_exact_sym_fcl,
            T_sym_expr=T_exact_sym_fcl,
            cl_sym_expr=cl_exact_sym_fcl,
            cd_sym_expr=cd_sym_zero_fcl,
            cs_sym_expr=cs_sym_zero_fcl,
            t_var=t_sym,
            x_var=x_sym,
            y_var=y_sym,
        )

        # Forcing terms dictionary (set fcl=0 for TE test)
        # Instantiate Field (uses numerical model methods)
        field = p1.SemiDiscreteField_CsTriple(
            grid=grid, model=fcl_test_model, forcing_terms=p1.NoForcingTerms(grid=grid)
        )

        # --- Evaluate numerical operator Fcl on exact state ---
        # 1. Get exact state StateVars using mms_spec (returns numerical arrays)
        exact_state = p1.StateVars(
            cp=mms_spec.cp(test_time_fcl, xx, yy),
            T=mms_spec.T(test_time_fcl, xx, yy),
            cl=mms_spec.cl(test_time_fcl, xx, yy),
            cd=mms_spec.cd(test_time_fcl, xx, yy),
            cs=mms_spec.cs(test_time_fcl, xx, yy),
            model=fcl_test_model,
            hh=grid.hh,
            kk=grid.kk,
        )

        # 2. Apply numerical Fcl operator
        Fcl_numerical_grid = field.Fcl(exact_state, test_time_fcl)

        # --- Get Analytical value of the spatial RHS operator (using corrected L_spatial_analytic_func_fcl) ---
        L_spatial_analytic_grid = L_spatial_analytic_func_fcl(
            test_time_fcl, grid.xx, grid.yy
        )

        # --- Calculate Truncation Error ---
        truncation_error_grid = Fcl_numerical_grid - L_spatial_analytic_grid

        # --- Compute H-norm of truncation error ---
        error_norm = grid.norm_H(truncation_error_grid) + np.finfo(float).eps
        print(f"    N={N}, H-Norm(Fcl Truncation Error) = {error_norm:.4e}")
        truncation_errors_H_norm.append(error_norm)

    # --- Calculate observed rates ---
    deliverable_from_test_caculate_observed_rates(
        name="H-Norm(Fcl Truncation)",
        errors=truncation_errors_H_norm,
        with_asserts=True
    )

    print("--- Fcl Symbolic Test Passed ---")


# Define test models in fixtures for better reuse if needed later
@pytest.fixture
def fcd_test_model():
    fcd_default_model_consts = p1.ModelConsts(
        R0=p1.R0,
        Ea=p1.Ea,
        K1=0.0,
        K2=0.0,
        K3=0.0,
        K4=0.0,
        DT=0.0,
        Dl_max=0.0,
        phi_l=0.0,
        gamma_T=0.0,
        Kd=1e-3,
        Sd=10.0,
        Dd_max=2.5e-6,
        phi_d=1e-5,
        phi_T=p1.Ea / p1.R0,
        r_sp=0.0,
    )
    # Return the instance which uses numerical methods internally
    fcd_default_model = p1.DefaultModel02(mc=fcd_default_model_consts)
    
    return fcd_default_model.copy()


# --- The Test Function ---
def test_Fcd_spatial_convergence_symbolic(fcd_test_model):  # Pass fixture
    """
    Verifies spatial convergence order of Fcd using MMSCaseSymbolic.
    Compares numerical operator Fcd applied to exact state vs analytical spatial RHS.
    """

    # --- Setup for Fcd Test ---
    # MMS solutions (use previous definitions or define new ones)
    cd_exact_sym_fcd = (
        sympy.sin(2 * pi_sym * x_sym) * sympy.sin(pi_sym * y_sym) * (1 + 0.5 * t_sym)
    )
    cl_exact_sym_fcd = (
        sympy.sin(pi_sym * x_sym) * sympy.sin(2 * pi_sym * y_sym) * (1 + t_sym)
    )
    cs_exact_sym_fcd = (
        sympy.cos(0.5 * pi_sym * x_sym) * sympy.cos(0.5 * pi_sym * y_sym) * (2 + t_sym)
    )  # Non-zero inside
    cp_exact_sym_fcd = (
        sympy.sin(pi_sym * x_sym) ** 2 * sympy.sin(pi_sym * y_sym) ** 2 * (3 + t_sym)
    )  # Non-zero inside
    T_exact_sym_fcd = 300 * (
        1 + 0.1 * sympy.sin(pi_sym * x_sym) * sympy.sin(pi_sym * y_sym) * (1 + 0.1 * t_sym)
    )  # Keep T > 0

    # --- Analytical Operator Calculation for Fcd ---
    # L_spatial(cd, cl, cs, cp, T) = div(Dd(cp,T) grad(cd)) + Kd*(Sd-cd)*(cl+1)*cs

    # Need Dd expression defined using ABSTRACT symbols for differentiation, then substitute.
    cp_abs_fcd = sympy.Symbol("cp_abstract_fcd")
    T_abs_fcd = sympy.Symbol("T_abstract_fcd")

    # Symbolic Dd using abstract symbols
    Dd_abstract_expr = (
        fcd_test_model.Dd_max
        * sympy.exp(-fcd_test_model.phi_d * cp_abs_fcd)
        * sympy.exp(-fcd_test_model.phi_T / T_abs_fcd)
    )

    # Substitute MMS expressions into Dd expression
    sub_map_fcd = {cp_abs_fcd: cp_exact_sym_fcd, T_abs_fcd: T_exact_sym_fcd}
    Dd_on_MMS = Dd_abstract_expr.subs(sub_map_fcd)

    # Gradients of cd_exact
    dx_cd_sym = sympy.diff(cd_exact_sym_fcd, x_sym)
    dy_cd_sym = sympy.diff(cd_exact_sym_fcd, y_sym)

    # Diffusion term: div(Dd grad(cd))
    diffusion_term_sym_fcd = sympy.diff(Dd_on_MMS * dx_cd_sym, x_sym) + sympy.diff(
        Dd_on_MMS * dy_cd_sym, y_sym
    )

    # Reaction term: Kd*(Sd-cd)*(cl+1)*cs
    reaction_term_sym_fcd = (
        fcd_test_model.Kd
        * (fcd_test_model.Sd - cd_exact_sym_fcd)
        * (cl_exact_sym_fcd + 1)
        * cs_exact_sym_fcd
    )

    # Full analytical spatial RHS
    L_spatial_analytic_sym_fcd = diffusion_term_sym_fcd + reaction_term_sym_fcd

    # Lambdify
    L_spatial_analytic_func_fcd = sympy.lambdify(
        (t_sym, x_sym, y_sym), L_spatial_analytic_sym_fcd, "numpy"
    )


    # --- Test Parameters ---
    N_list_fcd_sym = [8, 16, 32, 64]
    test_time_fcd = 0.1  # Use a non-zero time

    truncation_errors_H_norm = []

    print("\n--- Running Fcd Spatial Conv Test (using MMSCaseSymbolic) ---")
    for N in N_list_fcd_sym:
        print(f"  Testing N={N}...")
        grid = p1.make_uniform_grid(N, N)
        xx, yy = grid.xx, grid.yy

        # --- Instantiate MMS Spec using MMSCaseSymbolic ---
        mms_spec = p1.MMSCaseSymbolic(
            grid=grid,
            model=fcd_test_model,  # Use fixture
            cp_sym_expr=cp_exact_sym_fcd,
            T_sym_expr=T_exact_sym_fcd,
            cl_sym_expr=cl_exact_sym_fcd,
            cd_sym_expr=cd_exact_sym_fcd,
            cs_sym_expr=cs_exact_sym_fcd,
            t_var=t_sym,
            x_var=x_sym,
            y_var=y_sym,
        )

        # Zero forcing terms dict
        # Instantiate Field (uses numerical model methods)
        field = p1.SemiDiscreteField_CsTriple(
            grid=grid, model=fcd_test_model, forcing_terms=p1.NoForcingTerms(grid=grid)
        )

        # --- Evaluate numerical operator Fcd on exact state ---
        # 1. Get exact state StateVars at test_time
        gfp_exact_state = p1.StateVars(
            cp=mms_spec.cp(test_time_fcd, xx, yy),
            T=mms_spec.T(test_time_fcd, xx, yy),
            cl=mms_spec.cl(test_time_fcd, xx, yy),
            cd=mms_spec.cd(test_time_fcd, xx, yy),
            cs=mms_spec.cs(test_time_fcd, xx, yy),
            model=fcd_test_model,
            hh=grid.hh,
            kk=grid.kk,
        )

        # 2. Apply numerical Fcd operator
        # Since field forcing['fcd'] = 0, this IS the numerical spatial operator
        Fcd_numerical_grid = field.Fcd(gfp_exact_state, test_time_fcd)

        # --- Get Analytical value of the spatial RHS operator ---
        L_spatial_analytic_grid = L_spatial_analytic_func_fcd(
            test_time_fcd, grid.xx, grid.yy
        )

        # --- Calculate Truncation Error ---
        truncation_error_grid = Fcd_numerical_grid - L_spatial_analytic_grid

        # --- Compute H-norm of truncation error ---
        error_norm = grid.norm_H(truncation_error_grid) + np.finfo(float).eps
        print(f"    N={N}, H-Norm(Fcd Truncation Error) = {error_norm:.4e}")
        truncation_errors_H_norm.append(error_norm)

    # --- Calculate observed rates ---
    deliverable_from_test_caculate_observed_rates(
        name="H-Norm(Fcd Truncation)", errors=truncation_errors_H_norm, with_asserts=True
    )

    print("--- Fcd Symbolic Test Passed ---")


def test_T_cl_coupled_spatial_convergence(cvg_test_Tf: float, h_to_dt_const: float, spatial_refinements_N_list: List[int]):
    """
    Verifies spatial convergence for the coupled T-cl system using the full
    time integrator and an MMS setup via MMSCaseSymbolic.
    """

    # --- MMS for Coupled T-cl Test ---
    # Using different simple solutions for T and cl
    T_exact_sym_tc = (
        (1 - x_sym) * x_sym * (1 - y_sym) * y_sym * (1 / (1 + t_sym))
    )
    cl_exact_sym_tc = (
        (1 - x_sym) * x_sym * (1 - y_sym) * y_sym * (1 / (1 + t_sym))
    )
    # Set others exactly to zero
    cp_sym_zero_tc = sympy.S(0)
    cd_sym_zero_tc = sympy.S(0)
    cs_sym_zero_tc = sympy.S(0)

    # --- Test Model for Coupled T-cl ---
    tc_test_model_consts = p1.ModelConsts(
        R0=p1.R0,
        Ea=p1.Ea,
        K1=0.0,
        K2=0.0,
        K3=1e-3,  # <<< Couple T to FT
        K4=5e-3,  # <<< Couple cl to Fcl (cp=0 here)
        DT=1e-3,  # <<< Non-zero diffusion for T
        Dl_max=8.0e-4,  # <<< Non-zero diffusion for cl
        phi_l=0.0,  # <<< Simplest Dl (constant)
        gamma_T=1e-9,  # <<< Couple T to Fcl (via V1)
        Kd=0.0,
        Sd=0.0,
        Dd_max=0.0,
        phi_d=0.0,
        phi_T=0.0,
        r_sp=0.0,
    )
    tc_test_model = p1.DefaultModel02(mc=tc_test_model_consts)

    final_errors_T_H_norm = []
    final_errors_cl_H_norm = []

    class the_mms_case_cls(p1.MMSCaseSymbolic):
        def __init__(self, *, grid: p1.Grid, model: p1.DefaultModel01):
            super().__init__(
                grid=grid,
                model=model,
                cp_sym_expr=cp_sym_zero_tc,
                T_sym_expr=T_exact_sym_tc,
                cl_sym_expr=cl_exact_sym_tc,
                cd_sym_expr=cd_sym_zero_tc,
                cs_sym_expr=cs_sym_zero_tc,
            )

    print(
        "\n--- Running Coupled T-cl Spatial Conv Test (Symbolic MMS, Full Integrator) ---"
    )

    for N in spatial_refinements_N_list:
        print(f"  Testing N={N}...")
        h = 1.0/ N  # Uniform grid spacing
        dt_for_N = h_to_dt_const * (h ** 1.5)  # temporal error should be order h^3 then.
        print(f"    h={h:.4e}")
        print(f"    dt_for_N={dt_for_N:.4e} (h^2 * {h_to_dt_const})")
        grid = p1.make_uniform_grid(N, N)
        mms_trial = mms_utils.MMSTrial(
            grid=grid,
            model=tc_test_model,
            mms_case_cls=the_mms_case_cls,
            field_cls=p1.SemiDiscreteField_CsTriple,
            forcing_terms_cls=p1.ForcingTerms_CsTriple,
            integrator_cls=p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_CsTriple,
        )
        error_summary = mms_trial.run_for_errors(Tf=cvg_test_Tf, dt=dt_for_N)
        error_norm_T = error_summary.per_variable_sup_errors["T"]
        error_norm_cl = error_summary.per_variable_sup_errors["cl"]

        print(f"    N={N}, Final H-Norm Error(T) = {error_norm_T:.4e}")
        print(f"    N={N}, Final H-Norm Error(cl) = {error_norm_cl:.4e}")
        final_errors_T_H_norm.append(error_norm_T)
        final_errors_cl_H_norm.append(error_norm_cl)

    deliverable_from_test_caculate_observed_rates(
        name="T", errors=final_errors_T_H_norm, with_asserts=True
    )
    deliverable_from_test_caculate_observed_rates(
        name="cl", errors=final_errors_cl_H_norm, with_asserts=True
    )

    print("\n--- Coupled T-cl Spatial Convergence Test Passed ---")


def test_T_cl_cd_coupled_spatial_convergence(cvg_test_Tf: float, h_to_dt_const: float, spatial_refinements_N_list: List[int]):
    """
    Verifies spatial convergence for the coupled T-cl-cd system using the full
    time integrator and an MMS setup via MMSCaseSymbolic.
    """
    # --- MMS for Coupled T-cl-cd Test ---
    # Keep solutions non-trivial and compatible with zero BCs
    T_exact_sym_tcd = (
        5 * sympy.sin(pi_sym * x_sym) * sympy.sin(2 * pi_sym * y_sym) * (1 + t_sym)
    )
    cl_exact_sym_tcd = (sympy.sin(pi_sym * x_sym) * sympy.sin(pi_sym * y_sym) * t_sym) ** 2
    cd_exact_sym_tcd = sympy.sin(2 * pi_sym * x_sym) * sympy.sin(pi_sym * y_sym) * t_sym
    # Set cp and cs exactly to zero
    cp_sym_zero_tcd = sympy.S(0)
    cs_sym_zero_tcd = sympy.S(0)

    # --- Test Model for Coupled T-cl-cd ---
    # Activate constants relevant for FT, Fcl, Fcd coupling T, cl, cd
    # Note: K3 term in FT is zero because cp=0. V1 affects Fcl. Dd depends on T (cp=0). Kd, Sd affect Fcd.
    tcd_test_model_consts = p1.ModelConsts(
        R0=p1.R0,
        Ea=p1.Ea,
        K1=0.0,
        K2=0.0,
        K3=0.0,  # K3=0 ok as cp=0 anyway
        K4=1e-3,  # Coupling in Fcl
        DT=1e-3,  # T Diffusion
        Dl_max=8.0e-4,  # cl Diffusion
        phi_l=0.0,  # Constant Dl
        gamma_T=1e-9,  # V1 -> Fcl depends on T
        Kd=5e-4,  # Reaction term in Fcd
        Sd=10.0,  # Reaction term in Fcd
        Dd_max=2.5e-6,  # cd Diffusion
        phi_d=0.0,  # Dd only depends on T here
        phi_T=p1.Ea / p1.R0,  # Dd depends on T
        r_sp=0.0,
    )
    tcd_test_model = p1.DefaultModel02(mc=tcd_test_model_consts)


    final_errors_T_H_norm = []
    final_errors_cl_H_norm = []
    final_errors_cd_H_norm = []

    class the_mms_case_cls(p1.MMSCaseSymbolic):
        def __init__(self, *, grid: p1.Grid, model: p1.DefaultModel01):
            super().__init__(
                grid=grid,
                model=model,
                cp_sym_expr=cp_sym_zero_tcd,
                T_sym_expr=T_exact_sym_tcd,
                cl_sym_expr=cl_exact_sym_tcd,
                cd_sym_expr=cd_exact_sym_tcd,
                cs_sym_expr=cs_sym_zero_tcd,
            )

    print(
        "\n--- Running Coupled T-cl-cd Spatial Conv Test (Symbolic MMS, PC Integrator) ---"
    )

    for N in spatial_refinements_N_list:
        print(f"  Testing N={N}...")
        grid = p1.make_uniform_grid(N, N)
        h = 1.0 / N  # Uniform grid spacing
        dt_for_N = h_to_dt_const * (h ** 1.5) # temporal convergence is order 2, which means dt error should be on h ** 3.
        print(f"    h={h:.4e}")
        print(f"    dt_for_N={dt_for_N:.4e} (h^2 * {h_to_dt_const})")
        mms_trial = mms_utils.MMSTrial(
            grid=grid,
            model=tcd_test_model,
            mms_case_cls=the_mms_case_cls,
            field_cls=p1.SemiDiscreteField_CsTriple,
            forcing_terms_cls=p1.ForcingTerms_CsTriple,
            integrator_cls=p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_CsTriple,
        )
        error_summary = mms_trial.run_for_errors(Tf=cvg_test_Tf, dt=dt_for_N)
        error_norm_T = error_summary.per_variable_sup_errors["T"]
        error_norm_cl = error_summary.per_variable_sup_errors["cl"]
        error_norm_cd = error_summary.per_variable_sup_errors["cd"]

        print(f"    N={N}, Final H-Norm Error(T)  = {error_norm_T:.4e}")
        print(f"    N={N}, Final H-Norm Error(cl) = {error_norm_cl:.4e}")
        print(f"    N={N}, Final H-Norm Error(cd) = {error_norm_cd:.4e}")
        final_errors_T_H_norm.append(error_norm_T)
        final_errors_cl_H_norm.append(error_norm_cl)
        final_errors_cd_H_norm.append(error_norm_cd)

    deliverable_from_test_caculate_observed_rates(
        name="T", errors=final_errors_T_H_norm, with_asserts=True
    )
    deliverable_from_test_caculate_observed_rates(
        name="cl", errors=final_errors_cl_H_norm, with_asserts=True
    )
    deliverable_from_test_caculate_observed_rates(
        name="cd", errors=final_errors_cd_H_norm, with_asserts=True
    )

    print("\n--- Coupled T-cl-cd Spatial Convergence Test Passed ---")

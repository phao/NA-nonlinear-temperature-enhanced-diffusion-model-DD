import pytest
import numpy as np
import math

import prob1base as p1
import prob1_mms_cases as p1mc
from utils_for_testing import observed_rates_report


@pytest.fixture
def p1_modified_model_consts() -> p1.ModelConsts:
    """Provides prob1base.ModelConsts with reduced D_T."""
    consts_dict = p1.default_model_consts._asdict()
    consts_dict["K1"] = 1e-2
    consts_dict["K2"] = 1e-2
    consts_dict["K3"] = 1e-2
    consts_dict["K4"] = 1e-2
    consts_dict["DT"] = 1e-3  # Modified D_T from original test
    consts_dict["Dl_max"] = 8.01e-4
    consts_dict["phi_l"] = 1e-5
    consts_dict["gamma_T"] = 1e-09
    consts_dict["Kd"] = 1e-8
    consts_dict["Sd"] = 10
    consts_dict["Dd_max"] = 2.46e-6
    consts_dict["phi_d"] = 1e-5
    consts_dict["r_sp"] = 5e-2
    return p1.ModelConsts(**consts_dict)


@pytest.fixture
def corrector_test_setup(p1_modified_model_consts: p1.ModelConsts):
    """
    Sets up the necessary components from prob1base.py for testing corrector steps.
    """
    N_grid = 30
    M_grid = 30

    grid = p1.make_uniform_grid(N_grid, M_grid)
    model = p1.DefaultModel01(mc=p1_modified_model_consts)
    mms_case = p1mc.MMSCaseExpSin(grid=grid, model=model)
    forcing_terms_calculator = p1.ForcingTerms_CsTriple(mms_case=mms_case, model=model)
    semi_discrete_field = p1.SemiDiscreteField_CsTriple(
        grid=grid, model=model, forcing_terms=forcing_terms_calculator
    )

    time_integrator = p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_CsTriple(
        semi_discrete_field=semi_discrete_field, num_pc_steps=1, num_newton_steps=1
    )

    return grid, model, mms_case, time_integrator


def test_isolated_corrector_temporal_convergence_p1base_v2(corrector_test_setup):
    """
    Verifies the temporal convergence of the corrector steps for Cp and Cs
    in isolation, using exact values for T, Cl, Cd at t_{n+1}, directly
    using prob1base.py components. Corrected forcing term setup.
    """
    grid, model, mms_case, time_integrator = corrector_test_setup

    Tf = 2.0
    num_refinements = 5
    S_base = 10

    errors = {"cp": [], "cs": []}
    dts = []
    var_names = ["cp", "cs"]

    xx, yy = grid.xx, grid.yy

    print(
        "\nTemporal Convergence Test for Isolated Correctors (Cp, Cs) using prob1base (v2):"
    )
    for k in range(num_refinements):
        S_current = S_base * (2**k)
        dt = Tf / S_current
        dts.append(dt)

        t_n = 0.0
        t_n1 = dt
        print(f"Running with S_current={S_current}, dt={dt:.3e}")

        cp_n_exact_vals = mms_case.cp(t_n, xx, yy)
        T_n_exact_vals = mms_case.T(t_n, xx, yy)
        cl_n_exact_vals = mms_case.cl(t_n, xx, yy)
        cd_n_exact_vals = mms_case.cd(t_n, xx, yy)
        cs_n_exact_vals = mms_case.cs(t_n, xx, yy)

        state_at_t0 = p1.StateVars(
            cp_n_exact_vals,
            T_n_exact_vals,
            cl_n_exact_vals,
            cd_n_exact_vals,
            cs_n_exact_vals,
            model=model,
            hh=grid.hh,
            kk=grid.kk,
        )

        T_n1_exact_vals = mms_case.T(t_n1, xx, yy)
        cl_n1_exact_vals = mms_case.cl(t_n1, xx, yy)
        cd_n1_exact_vals = mms_case.cd(t_n1, xx, yy)

        cp_n1_computed = time_integrator.corrector_cp_step(
            T_n1_exact_vals,
            cl_n1_exact_vals,
            _cd1_ignroed=None,
            at_t0=state_at_t0,
            t0=t_n,
            dt=dt,
        )

        cs_n1_computed = time_integrator.corrector_cs_step(
            _T1_ignored=None,
            cl1=cl_n1_exact_vals,
            cd1=cd_n1_exact_vals,
            at_t0=state_at_t0,
            t0=t_n,
            dt=dt,
        )

        cp_n1_exact_vals = mms_case.cp(t_n1, xx, yy)
        cs_n1_exact_vals = mms_case.cs(t_n1, xx, yy)

        errors["cp"].append(grid.norm_H(cp_n1_computed - cp_n1_exact_vals))
        errors["cs"].append(grid.norm_H(cs_n1_computed - cs_n1_exact_vals))

        print(f"  Error(cp) = {errors['cp'][-1]:.4e}")
        print(f"  Error(cs) = {errors['cs'][-1]:.4e}")

    print("\nObserved Rates (Isolated Correctors - prob1base v2):")
    expected_order = 2.0  # Trapezoidal-based corrector
    tolerance = 0.1

    for var in var_names:
        print(f"  Variable {var}:")
        observed_rates_report(
            errors=errors[var],
            cmp_type="least",
            expected_rate=expected_order,
            tolerance=tolerance,
        )

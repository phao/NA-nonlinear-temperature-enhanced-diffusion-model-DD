# test_time_integrator_hcs_triple_full_step.py
import pytest
import numpy as np
from numpy.testing import assert_allclose
import sympy

import prob1base as p1  # Assuming prob1base.py is importable
from typing import Callable, Dict, Tuple, List, Any, Type

# --- Test-Specific Model Constants ---
# Using a distinct set for these integration tests
R0_test_fs: float = 8.3144621
Ea_test_fs: float = 1.60217662e-19
test_model_consts_fs: p1.ModelConsts = p1.ModelConsts(
    R0=R0_test_fs,
    Ea=Ea_test_fs,
    K1=1.22e-2,
    K2=1.32e-2,
    K3=1.42e-2,
    K4=1.52e-2,
    DT=1.22e-3,
    Dl_max=8.22e-4,
    phi_l=1.22e-5,
    gamma_T=1.22e-9,
    Kd=1.22e-8,
    Sd=15,
    Dd_max=2.62e-6,
    phi_d=1.22e-5,
    phi_T=Ea_test_fs / R0_test_fs,
    r_sp=5.22e-2,
    T_ref=303,
)


# --- Fixtures ---
@pytest.fixture
def grid_fixture_fs() -> p1.Grid:
    """Small grid for single step test."""
    return p1.make_uniform_grid(
        N=4, M=4
    )  # Reduced N, M for faster symbolic setup if it becomes an issue


@pytest.fixture
def model_fixture_fs() -> p1.DefaultModel01:
    """Model instance for full step tests."""
    return p1.DefaultModel01(mc=test_model_consts_fs)


@pytest.fixture
def mms_case_factory_fs(grid_fixture_fs: p1.Grid, model_fixture_fs: p1.DefaultModel01):
    """Factory to create MMSCaseSymbolic for single step integrator tests."""

    default_cs_spatial_profile = p1.x_sym * (1 - p1.x_sym) * p1.y_sym * (1 - p1.y_sym)
    # default_cs_spatial_profile = sympy.S(1) # For spatially constant Cs

    def _factory(
        cs_initial_val: float = 1.0,  # Initial value for Cs ramp at t=0
        cs_slope: float = -0.5,  # Slope for Cs ramp (A - B*t, so B = -cs_slope if A=cs_initial)
        spatial_profile_expr: sympy.Expr = default_cs_spatial_profile,
    ) -> p1.MMSCaseSymbolic:

        # Manufactured solutions:
        # Cp, T, Cl, Cd are constant zero in space and time.
        cp_ex_sym: sympy.Expr = sympy.S(0)
        T_ex_sym: sympy.Expr = sympy.S(0)
        cl_ex_sym: sympy.Expr = sympy.S(0)
        cd_ex_sym: sympy.Expr = sympy.S(0)

        # Cs is a linear ramp in time, multiplied by a spatial profile (or constant if profile is 1)
        # cs_exact(t,x,y) = (cs_initial_val + cs_slope * t) * spatial_profile(x,y)
        cs_t_ramp_sym: sympy.Expr = (
            sympy.Float(cs_initial_val) + sympy.Float(cs_slope) * p1.t_sym
        )
        cs_ex_sym: sympy.Expr = cs_t_ramp_sym * spatial_profile_expr
        # Ensure cs can go negative if slope and time dictate
        # No sympy.Max(..., 0) here, as per latest understanding for testing general integrator

        return p1.MMSCaseSymbolic(
            grid=grid_fixture_fs,
            model=model_fixture_fs,
            cp_sym_expr=cp_ex_sym,
            T_sym_expr=T_ex_sym,
            cl_sym_expr=cl_ex_sym,
            cd_sym_expr=cd_ex_sym,
            cs_sym_expr=cs_ex_sym,
            t_var=p1.t_sym,
            x_var=p1.x_sym,
            y_var=p1.y_sym,
        )

    return _factory


@pytest.fixture
def forcing_terms_fs(
    mms_case_factory_fs: Callable[..., p1.MMSCaseSymbolic],
    model_fixture_fs: p1.DefaultModel01,
) -> p1.ForcingTerms_HCsTriple:
    """
    Real ForcingTerms_HCsTriple based on an MMS where Cp,T,Cl,Cd are zero
    and Cs is a ramp.
    This specific instance will be configured by tests if parameters need to vary.
    """
    # Create a default MMS case for this fixture
    default_mms_case = mms_case_factory_fs()  # Uses default ramp for Cs
    return p1.ForcingTerms_HCsTriple(mms_case=default_mms_case, model=model_fixture_fs)


@pytest.fixture
def semi_discrete_field_fs(
    grid_fixture_fs: p1.Grid,
    model_fixture_fs: p1.DefaultModel01,
    forcing_terms_fs: p1.ForcingTerms_HCsTriple,  # Use the real forcing terms with MMS
) -> p1.SemiDiscreteField_HCsTriple:
    """Real SemiDiscreteField_HCsTriple for integration tests."""
    return p1.SemiDiscreteField_HCsTriple(
        grid=grid_fixture_fs, model=model_fixture_fs, forcing_terms=forcing_terms_fs
    )


@pytest.fixture
def integrator_fs(
    semi_discrete_field_fs: p1.SemiDiscreteField_HCsTriple,
) -> p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple:
    """Real HCsTriple integrator for full step tests."""
    return p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple(
        semi_discrete_field=semi_discrete_field_fs,
        num_pc_steps=1,  # Single PC iteration for simplicity in tracing
        num_newton_steps=1,  # Single Newton iteration
    )


# --- Single Step Test using MMS ---
@pytest.mark.parametrize(
    "cs_initial_val, cs_slope, dt, t0",
    [
        # Case 1: Cs starts positive, decreases but stays positive
        (1.0, -0.5, 0.01, 0.0),  # Cs(0)=1.0, Cs(0.01)=1-0.005 = 0.995
        # Case 2: Cs starts positive, ramp makes it zero at t0+dt
        (
            0.1,
            -10.0,
            0.01,
            0.0,
        ),  # Cs(0)=0.1, Cs_slope=-10. Cs(0.01) = 0.1 - 10*0.01 = 0.1 - 0.1 = 0.0
        # Case 3: Cs starts positive, ramp makes it negative at t0+dt
        (
            0.1,
            -15.0,
            0.01,
            0.0,
        ),  # Cs(0)=0.1, Cs_slope=-15. Cs(0.01) = 0.1 - 15*0.01 = 0.1 - 0.15 = -0.05
        # Case 4: Cs starts negative, stays negative
        (-0.2, -0.5, 0.01, 0.0),  # Cs(0)=-0.2, Cs(0.01)=-0.2-0.005 = -0.205
        # Case 5: Cs starts at zero, becomes negative
        (0.0, -5.0, 0.01, 0.0),  # Cs(0)=0, Cs(0.01) = -0.05
    ],
)
def test_integrator_single_step_with_mms_cs_ramp(
    integrator_fs: p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple,
    mms_case_factory_fs: Callable[..., p1.MMSCaseSymbolic],
    model_fixture_fs: p1.DefaultModel01,  # For re-initing field with new forcing
    grid_fixture_fs: p1.Grid,  # For re-initing field
    cs_initial_val: float,
    cs_slope: float,
    dt: float,
    t0: float,
):
    """
    Tests one full step of the integrator.
    MMS: Cp=T=Cl=Cd=0. Cs(t) = (cs_initial_val + cs_slope*t) * spatial_profile.
    The spatial_profile is x(1-x)y(1-y) by default in mms_case_factory_fs.
    This means cs is not spatially constant, which is a more rigorous test for forcing terms.
    """
    # 1. Create the specific MMS case for this test's parameters
    # Using default spatial profile x(1-x)y(1-y) for Cs
    current_mms_case: p1.MMSCaseSymbolic = mms_case_factory_fs(
        cs_initial_val=cs_initial_val, cs_slope=cs_slope
    )

    # 2. Create ForcingTerms for this MMS case
    current_forcing_terms: p1.ForcingTerms_HCsTriple = p1.ForcingTerms_HCsTriple(
        mms_case=current_mms_case, model=model_fixture_fs
    )

    # 3. Update the integrator's semi_discrete_field to use these new forcing terms
    # (The model and grid for the field remain the same as those used by the integrator)
    integrator_fs.semi_discrete_field._forcing_terms = current_forcing_terms
    # Re-assign forcing methods on the field instance
    for term_name in ["fcp", "fT", "fcl", "fcd", "fcs"]:
        setattr(
            integrator_fs.semi_discrete_field,
            term_name,
            getattr(current_forcing_terms, term_name),
        )

    # 4. Set Initial Condition from MMS at t0
    xx, yy = grid_fixture_fs.xx, grid_fixture_fs.yy
    initial_state_dict: Dict[str, np.ndarray] = {
        "cp": current_mms_case.cp(t0, xx, yy),
        "T": current_mms_case.T(t0, xx, yy),
        "cl": current_mms_case.cl(t0, xx, yy),
        "cd": current_mms_case.cd(t0, xx, yy),
        "cs": current_mms_case.cs(t0, xx, yy),
    }
    initial_state: p1.StateVars = p1.StateVars(
        **initial_state_dict,
        model=model_fixture_fs,
        hh=grid_fixture_fs.hh,
        kk=grid_fixture_fs.kk,
    )

    # 5. Perform one step
    final_state: p1.StateVars = integrator_fs.step(initial_state, t0=t0, dt=dt)
    t1: float = t0 + dt

    xx, yy = grid_fixture_fs.xx, grid_fixture_fs.yy

    # 6. Get Exact Solution at t1
    cp_exact_t1: np.ndarray = current_mms_case.cp(t1, xx, yy)
    T_exact_t1: np.ndarray = current_mms_case.T(t1, xx, yy)
    cl_exact_t1: np.ndarray = current_mms_case.cl(t1, xx, yy)
    cd_exact_t1: np.ndarray = current_mms_case.cd(t1, xx, yy)
    cs_exact_t1: np.ndarray = current_mms_case.cs(t1, xx, yy)

    # 7. Verification
    # Cp, T, Cl, Cd should be very close to their exact values (which are 0)
    # The atol needs to be loose enough to accommodate one step of a 2nd order method error (dt^2 or dt^3 for local error)
    # but tight enough to catch gross errors. For a single step, 1e-5 might be reasonable if dt=0.01.
    # For local error O(dt^3), error ~ C*dt^3. If dt=1e-2, dt^3=1e-6.
    # Let's start with a slightly looser tolerance and refine if needed.
    atol_non_cs: float = 1e-12
    rtol_non_cs: float = 1e-8

    assert_allclose(
        final_state.cp,
        cp_exact_t1,
        rtol=rtol_non_cs,
        atol=atol_non_cs,
        err_msg=f"Cp after 1 step mismatch at t={t1}",
    )
    assert_allclose(
        final_state.T,
        T_exact_t1,
        rtol=rtol_non_cs,
        atol=atol_non_cs,
        err_msg=f"T after 1 step mismatch at t={t1}",
    )
    assert_allclose(
        final_state.cl,
        cl_exact_t1,
        rtol=rtol_non_cs,
        atol=atol_non_cs,
        err_msg=f"Cl after 1 step mismatch at t={t1}",
    )
    assert_allclose(
        final_state.cd,
        cd_exact_t1,
        rtol=rtol_non_cs,
        atol=atol_non_cs,
        err_msg=f"Cd after 1 step mismatch at t={t1}",
    )

    # For Cs, the error will also be present.
    # The exact value cs_exact_t1 can be non-zero, so rtol is more relevant here.
    # It includes the spatial profile, so we compare arrays.
    atol_cs: float = 1e-12 
    rtol_cs: float = 1e-6

    # Debugging output
    # print(f"\nCs0={cs_initial_val}, Slope={cs_slope}, dt={dt}")
    # print(f"Expected Cs(t1) (interior mean): {np.mean(cs_exact_t1[1:-1,1:-1]):.4e}")
    # print(f"Actual Cs(t1) (interior mean):   {np.mean(final_state.cs[1:-1,1:-1]):.4e}")
    # print(f"Difference (interior mean):      {np.mean(final_state.cs[1:-1,1:-1] - cs_exact_t1[1:-1,1:-1]):.4e}")

    assert_allclose(
        final_state.cs,
        cs_exact_t1,
        rtol=rtol_cs,
        atol=atol_cs,
        err_msg=f"Cs after 1 step mismatch at t={t1}",
    )

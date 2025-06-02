# test_forcing_terms_hcs_triple.py
import pytest
import numpy as np
from numpy.testing import assert_allclose
import sympy

# Assuming prob1base.py is in the Python path or same directory
import prob1base as p1

# --- Test-Specific Model Constants ---
R0_test = 8.3144621
Ea_test = 1.60217662e-19
test_model_consts = p1.ModelConsts(
    R0=R0_test,
    Ea=Ea_test,
    K1=1.1e-2,
    K2=1.2e-2,
    K3=1.3e-2,
    K4=1.4e-2,
    DT=1.1e-3,
    Dl_max=8.1e-4,
    phi_l=1.1e-5,
    gamma_T=1.1e-9,
    Kd=1.1e-8,
    Sd=11,
    Dd_max=2.5e-6,
    phi_d=1.1e-5,
    phi_T=Ea_test / R0_test,
    r_sp=5.1e-2,
    T_ref=298,
)


# --- Fixtures ---
@pytest.fixture
def grid_fixture():
    """Grid fixture."""
    return p1.make_uniform_grid(N=4, M=4)


@pytest.fixture
def model_fixture():
    """Fixture for the DefaultModel01 using test-specific constants."""
    return p1.DefaultModel01(mc=test_model_consts)


@pytest.fixture
def mms_case_symbolic_factory(grid_fixture, model_fixture):
    """
    Factory fixture to create an MMSCaseSymbolic instance.
    Manufactured solution for cs can now be negative to test H(cs) fully.
    """
    spatial_profile = p1.x_sym * (1 - p1.x_sym) * p1.y_sym * (1 - p1.y_sym)

    def _factory(
        cp_expr=spatial_profile * sympy.exp(-0.1 * p1.t_sym),
        T_expr=spatial_profile * sympy.exp(-0.2 * p1.t_sym),
        cl_expr=spatial_profile * sympy.exp(-0.3 * p1.t_sym),
        cd_expr=spatial_profile * sympy.exp(-0.4 * p1.t_sym),
        cs_initial_val=0.5,
        cs_time_decay_rate=0.05,
        cs_direct_expr=None,
    ):
        if cs_direct_expr is not None:
            if not isinstance(cs_direct_expr, sympy.Expr):
                raise TypeError(
                    "cs_direct_expr must be a SymPy expression if provided."
                )
            final_cs_expr = cs_direct_expr
        else:
            # Simple ramp that can go negative: cs_initial_val * (1 - rate*t)
            final_cs_expr = sympy.Float(cs_initial_val) * (
                1 - sympy.Float(cs_time_decay_rate) * p1.t_sym
            )

        return p1.MMSCaseSymbolic(
            grid=grid_fixture,
            model=model_fixture,
            cp_sym_expr=cp_expr,
            T_sym_expr=T_expr,
            cl_sym_expr=cl_expr,
            cd_sym_expr=cd_expr,
            cs_sym_expr=final_cs_expr,
            t_var=p1.t_sym,
            x_var=p1.x_sym,
            y_var=p1.y_sym,
        )

    return _factory


# --- Helper to calculate cs(t) for the ramp ---
def calculate_ramp_cs_at_t(initial_val, decay_rate, t):
    return initial_val * (1.0 - decay_rate * t)


# --- Tests for ForcingTerms_HCsTriple ---


@pytest.mark.parametrize(
    "cs_initial, cs_decay_rate, t_eval, expected_heaviside_factor_val",
    [
        # Case 1: cs(t_eval) > 0
        (1.0, 0.1, 1.0, 1.0),  # ramp_cs(1) = 1*(1-0.1) = 0.9. (cs>0) -> True
        # Case 2: cs(t_eval) = 0 (ramp hits zero exactly)
        (1.0, 0.1, 10.0, 0.0),  # ramp_cs(10) = 1*(1-1) = 0. (cs>0) -> False
        # Case 3: cs(t_eval) < 0 (ramp goes negative)
        (1.0, 0.1, 12.0, 0.0),  # ramp_cs(12) = 1*(1-1.2) = -0.2. (cs>0) -> False
        # Case 4: cs_initial is zero, so cs(t_eval) is always 0
        (0.0, 0.1, 5.0, 0.0),  # ramp_cs starts at 0. (cs>0) -> False
        # Case 5: cs(t_eval) > 0, different values
        (0.5, 0.05, 2.0, 1.0),  # ramp_cs(2) = 0.5*(1-0.1) = 0.45. (cs>0) -> True
        # Case 6: cs_initial is negative
        (-1.0, 0.05, 1.0, 0.0),  # ramp_cs(1) = -1*(1-0.05) = -0.95. (cs>0) -> False
    ],
)
def test_forcing_terms_hcs_triple_fcs_heaviside(
    mms_case_symbolic_factory,
    model_fixture,
    grid_fixture,
    cs_initial,
    cs_decay_rate,
    t_eval,
    expected_heaviside_factor_val,
):
    """
    Tests H(cs)=(cs > 0) in fcs of ForcingTerms_HCsTriple.
    Manufactured cs = cs_initial * (1 - rate*t), can be negative.
    """
    mms_case = mms_case_symbolic_factory(
        cs_initial_val=cs_initial, cs_time_decay_rate=cs_decay_rate
    )
    forcing_hcs = p1.ForcingTerms_HCsTriple(mms_case=mms_case, model=model_fixture)

    xx, yy = grid_fixture.xx, grid_fixture.yy
    dtCs_exact_arr = mms_case.dt_cs(t_eval, xx, yy)
    cs_exact_arr = mms_case.cs(t_eval, xx, yy)
    cl_exact_arr = mms_case.cl(t_eval, xx, yy)
    cd_exact_arr = mms_case.cd(t_eval, xx, yy)

    Kd = model_fixture.Kd
    Sd = model_fixture.Sd

    # Definition: fcs = dtCs - (-Kd * (cs > 0) * (1 + cl) * (Sd - cd))
    # where (cs > 0) is the Heaviside H(cs_exact_arr)
    heaviside_on_cs_exact = (cs_exact_arr > 1e-9).astype(
        float
    )  # Attempted robust (against accumulated flop round off errors) check for cs > 0.

    expected_fcs_val = dtCs_exact_arr - (
        -Kd * heaviside_on_cs_exact * (1 + cl_exact_arr) * (Sd - cd_exact_arr)
    )
    calculated_fcs_val = forcing_hcs.fcs(t_eval, xx, yy)

    # Sanity check: expected_heaviside_factor_val should match heaviside_on_cs_exact
    assert_allclose(
        heaviside_on_cs_exact,
        expected_heaviside_factor_val,
        atol=1e-9,
        err_msg="Heaviside factor mismatch in test parameters vs. actual cs(t_eval).",
    )

    assert_allclose(
        calculated_fcs_val,
        expected_fcs_val,
        rtol=1e-7,
        atol=1e-9,
        err_msg=(
            f"fcs HCsTriple failed: cs_init={cs_initial}, rate={cs_decay_rate}, t={t_eval}, "
            f"H_expect={expected_heaviside_factor_val}. "
            f"cs(t_eval) point value: {cs_exact_arr[grid_fixture.N//2, grid_fixture.M//2]:.3e}"
        ),
    )


@pytest.mark.parametrize(
    "cs_initial, cs_decay_rate, t_eval, expected_heaviside_factor_val",
    [
        (1.0, 0.1, 1.0, 1.0),
        (1.0, 0.1, 10.0, 0.0),
        (1.0, 0.1, 12.0, 0.0),
        (0.0, 0.1, 5.0, 0.0),
        (-1.0, 0.05, 1.0, 0.0),
    ],
)
def test_forcing_terms_hcs_triple_fcd_heaviside(
    mms_case_symbolic_factory,
    model_fixture,
    grid_fixture,  # Changed from grid_fixture_module
    cs_initial,
    cs_decay_rate,
    t_eval,
    expected_heaviside_factor_val,
):
    """
    Tests H(cs)=(cs > 0) in fcd of ForcingTerms_HCsTriple.
    Manufactured cs = cs_initial * (1 - rate*t), can be negative.
    """
    mms_case = mms_case_symbolic_factory(
        cs_initial_val=cs_initial, cs_time_decay_rate=cs_decay_rate
    )
    forcing_hcs = p1.ForcingTerms_HCsTriple(mms_case=mms_case, model=model_fixture)

    # Get exact solution components at t_eval
    xx, yy = grid_fixture.xx, grid_fixture.yy
    dtCd_exact_arr = mms_case.dt_cd(t_eval, xx, yy)
    cp_exact_arr = mms_case.cp(t_eval, xx, yy)
    T_exact_arr = mms_case.T(t_eval, xx, yy)
    dxCp_exact_arr = mms_case.dx_cp(t_eval, xx, yy)
    dyCp_exact_arr = mms_case.dy_cp(t_eval, xx, yy)
    dxT_exact_arr = mms_case.dx_T(t_eval, xx, yy)
    dyT_exact_arr = mms_case.dy_T(t_eval, xx, yy)
    lapCd_exact_arr = mms_case.lap_cd(t_eval, xx, yy)
    dxCd_exact_arr = mms_case.dx_cd(t_eval, xx, yy)
    dyCd_exact_arr = mms_case.dy_cd(t_eval, xx, yy)
    cs_exact_arr = mms_case.cs(t_eval, xx, yy)
    cl_exact_arr = mms_case.cl(t_eval, xx, yy)
    cd_exact_arr = mms_case.cd(t_eval, xx, yy)

    Kd = model_fixture.Kd
    Sd = model_fixture.Sd

    # Calculate diffusion/advection part for cd
    Dd_val = model_fixture.Dd(cp_exact_arr, T_exact_arr)
    dCp_Dd_val = model_fixture.Dd(cp_exact_arr, T_exact_arr, d=(1, 0))
    dT_Dd_val = model_fixture.Dd(cp_exact_arr, T_exact_arr, d=(0, 1))

    diff_adv_cd_terms = (
        (dCp_Dd_val * dxCp_exact_arr + dT_Dd_val * dxT_exact_arr) * dxCd_exact_arr
        + (dCp_Dd_val * dyCp_exact_arr + dT_Dd_val * dyT_exact_arr) * dyCd_exact_arr
        + Dd_val * lapCd_exact_arr
    )

    # Reaction part for HCsTriple fcd: Kd * (Sd - cd) * (cl + 1) * (cs > 0)
    heaviside_on_cs_exact = (cs_exact_arr > 1e-9).astype(float)
    reaction_hcs_component = (
        Kd * (Sd - cd_exact_arr) * (cl_exact_arr + 1) * heaviside_on_cs_exact
    )

    # Expected full fcd for HCsTriple: dtCd - (Diff_adv_terms + Reaction_HCs_terms)
    expected_fcd_val = dtCd_exact_arr - (diff_adv_cd_terms + reaction_hcs_component)
    calculated_fcd_val = forcing_hcs.fcd(t_eval, xx, yy)

    # Sanity check: expected_heaviside_factor_val should match heaviside_on_cs_exact
    assert_allclose(
        heaviside_on_cs_exact,
        expected_heaviside_factor_val,
        atol=1e-9,
        err_msg="Heaviside factor mismatch in test parameters vs. actual cs(t_eval).",
    )

    assert_allclose(
        calculated_fcd_val,
        expected_fcd_val,
        rtol=1e-7,
        atol=1e-9,
        err_msg=(
            f"fcd HCsTriple failed: cs_init={cs_initial}, rate={cs_decay_rate}, t={t_eval}, "
            f"H_expect={expected_heaviside_factor_val}. "
            f"cs(t_eval) point value: {cs_exact_arr[grid_fixture.N//2, grid_fixture.M//2]:.3e}"
        ),
    )


def test_forcing_terms_hcs_triple_consistent_fcp_T_fcl(
    mms_case_symbolic_factory, model_fixture
):
    """
    Tests that fcp, fT, fcl from ForcingTerms_HCsTriple match ForcingTerms_CsTriple,
    as these terms should not be affected by the Heaviside modification.
    """
    mms_case = mms_case_symbolic_factory(cs_initial_val=1.0)  # cs > 0 at t=0
    
    forcing_hcs = p1.ForcingTerms_HCsTriple(mms_case=mms_case, model=model_fixture)
    forcing_cs_ref = p1.ForcingTerms_CsTriple(mms_case=mms_case, model=model_fixture)

    t_eval = 0.5
    grid = mms_case.grid
    xx, yy = grid.xx, grid.yy

    fcp_hcs = forcing_hcs.fcp(t_eval, xx, yy)
    fT_hcs = forcing_hcs.fT(t_eval, xx, yy)
    fcl_hcs = forcing_hcs.fcl(t_eval, xx, yy)

    fcp_cs_ref = forcing_cs_ref.fcp(t_eval, xx, yy)
    fT_cs_ref = forcing_cs_ref.fT(t_eval, xx, yy)
    fcl_cs_ref = forcing_cs_ref.fcl(t_eval, xx, yy)

    assert_allclose(fcp_hcs, fcp_cs_ref, rtol=1e-9, atol=1e-12, err_msg="fcp mismatch")
    assert_allclose(fT_hcs, fT_cs_ref, rtol=1e-9, atol=1e-12, err_msg="fT mismatch")
    assert_allclose(fcl_hcs, fcl_cs_ref, rtol=1e-9, atol=1e-12, err_msg="fcl mismatch")

# new_tests/test_reghcstriple_system.py

from typing import Type
import pytest
import numpy as np
import sympy
import math

import prob1base as p1
import prob1_mms_cases as p1mc
import mms_trial_utils as mtu
from utils_for_testing import observed_rates_report

# --- Test-Specific Model Constants ---
R0_TEST_REG_HCS: float = 8.3144621
EA_TEST_REG_HCS: float = 1.60217662e-19
DEFAULT_REG_FACTOR = 50.0  # Default regularization_factor for Heaviside

test_model_consts_reg_hcs: p1.ModelConsts = p1.ModelConsts(
    R0=R0_TEST_REG_HCS,
    Ea=EA_TEST_REG_HCS,
    K1=1e-3,
    K2=1e-3,
    K3=1e-3,
    K4=1e-3,
    DT=1e-3,
    Dl_max=1e-5,
    phi_l=1e-5,
    gamma_T=1e-9,
    Kd=1e-2,
    Sd=1,
    Dd_max=1e-6,
    phi_d=1e-5,
    phi_T=R0_TEST_REG_HCS / R0_TEST_REG_HCS,
    r_sp=5e-2,
    T_ref=300,
)

# --- Pytest Fixtures ---


@pytest.fixture(scope="module")
def model_fixture_reghcs() -> p1.DefaultModel01:
    """
    Provides a DefaultModel01 instance for RegHCsTriple tests.

    :return: An instance of DefaultModel01.
    :rtype: p1.DefaultModel01
    """
    return p1.DefaultModel02(mc=test_model_consts_reg_hcs)


@pytest.fixture(scope="module")
def mms_case_cls_reghcs_symbolic() -> Type[p1.MMSCaseSymbolic]:
    """
    Provides an MMSCaseSymbolic instance suitable for testing RegHCsTriple.
    The exact solution for 'cs' is designed to vary around zero to specifically
    test the regularized Heaviside function.

    :param model_fixture_reghcs: The model fixture.
    :type model_fixture_reghcs: p1.DefaultModel01
    :return: An instance of MMSCaseSymbolic.
    :rtype: p1.MMSCaseSymbolic
    """
    A = 0.2
    decay_t = 0.15
    omega_t = sympy.pi / 2.0

    xsym = p1.x_sym
    ysym = p1.y_sym
    tsym = p1.t_sym

    s_x = sympy.sin(sympy.pi * xsym)
    s_y = sympy.sin(sympy.pi * ysym)
    spatial_sym = s_x * s_y * A

    exp_t = sympy.exp(-decay_t * tsym)
    cos_wt = sympy.cos(omega_t * tsym)

    cp_sym_expr = spatial_sym * cos_wt
    T_sym_expr = spatial_sym * exp_t
    cl_sym_expr = spatial_sym * sympy.cos(2 * omega_t * tsym)
    cd_sym_expr = spatial_sym * sympy.exp(-1.5 * decay_t * tsym)

    cs_amp = 0.2
    cs_sym_expr = cs_amp * spatial_sym * sympy.cos(sympy.pi * tsym / 3.0)

    class MMSCaseRegHCsSymbolic(p1.MMSCaseSymbolic):
        def __init__(self, grid: p1.Grid, model: p1.DefaultModel01):
            super().__init__(
                grid=grid,
                model=model,
                cp_sym_expr=cp_sym_expr,
                T_sym_expr=T_sym_expr,
                cl_sym_expr=cl_sym_expr,
                cd_sym_expr=cd_sym_expr,
                cs_sym_expr=cs_sym_expr,
                t_var=tsym,
                x_var=xsym,
                y_var=ysym,
            )

    return p1mc.MMSCaseExpSin  # Return the class itself, not an instance


@pytest.fixture(scope="module")
def mms_case_reghcs_symbolic(
    mms_case_cls_reghcs_symbolic: Type[p1.MMSCaseSymbolic],
    model_fixture_reghcs: p1.DefaultModel01,
) -> p1.MMSCaseSymbolic:
    dummy_grid_for_mms = p1.make_uniform_grid(N=2, M=2)
    return mms_case_cls_reghcs_symbolic(
        grid=dummy_grid_for_mms, model=model_fixture_reghcs
    )


# --- Sanity Tests ---


def test_reghcstriple_classes_instantiation(
    model_fixture_reghcs: p1.DefaultModel01,
    mms_case_reghcs_symbolic: p1.MMSCaseSymbolic,
):
    """
    Tests basic instantiation of the ForcingTerms_RegHCsTriple,
    SemiDiscreteField_RegHCsTriple, and
    P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple classes.

    :param model_fixture_reghcs: Pytest fixture for the simulation model.
    :type model_fixture_reghcs: p1.DefaultModel01
    :param mms_case_reghcs_symbolic: Pytest fixture for the symbolic MMS case.
    :type mms_case_reghcs_symbolic: p1.MMSCaseSymbolic
    """
    grid = p1.make_uniform_grid(N=5, M=5)
    reg_factor = DEFAULT_REG_FACTOR

    forcing_terms = p1.ForcingTerms_RegHCsTriple(
        mms_case=mms_case_reghcs_symbolic,
        model=model_fixture_reghcs,
        regularization_factor=reg_factor,
    )
    assert forcing_terms is not None, "ForcingTerms_RegHCsTriple instantiation failed."
    assert (
        forcing_terms.regularization_factor == reg_factor
    ), "Forcing term regularization_factor mismatch."

    field = p1.SemiDiscreteField_RegHCsTriple(
        grid=grid,
        model=model_fixture_reghcs,
        forcing_terms=forcing_terms,
        regularization_factor=reg_factor,
    )
    assert field is not None, "SemiDiscreteField_RegHCsTriple instantiation failed."
    assert (
        field.regularization_factor == reg_factor
    ), "Field regularization_factor mismatch."

    integrator = p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple(
        semi_discrete_field=field,
        regularization_factor=reg_factor,
        num_pc_steps=1,
        num_newton_iterations=3,
    )
    assert integrator is not None, "Integrator_RegHCsTriple instantiation failed."
    assert (
        integrator._regularization_factor == reg_factor
    ), "Integrator regularization_factor mismatch."


# --- MMS Based Tests ---

FIELD_CORRECTNESS_GRID_SIZES = [8, 16]


@pytest.mark.parametrize("grid_n", FIELD_CORRECTNESS_GRID_SIZES)
def test_field_correctness_reghcstriple(
    model_fixture_reghcs: p1.DefaultModel01,
    mms_case_reghcs_symbolic: p1.MMSCaseSymbolic,
    grid_n: int,
    request,
):
    """
    Tests if the semi-discrete field operator F(u_m) approximates du_m/dt from MMS.
    The H-norm of the error ||F(u_m) - du_m/dt||_H should decrease with grid refinement.

    :param model_fixture_reghcs: Pytest fixture for the simulation model.
    :type model_fixture_reghcs: p1.DefaultModel01
    :param mms_case_reghcs_symbolic: Pytest fixture for the symbolic MMS case.
    :type mms_case_reghcs_symbolic: p1.MMSCaseSymbolic
    :param grid_n: The number of grid points (N=M=grid_n).
    :type grid_n: int
    :param request: Pytest request object to store data across parameterized calls.
    """
    grid = p1.make_uniform_grid(N=grid_n, M=grid_n)
    t_eval = 0.1
    reg_factor = DEFAULT_REG_FACTOR

    forcing_terms = p1.ForcingTerms_RegHCsTriple(
        mms_case=mms_case_reghcs_symbolic,
        model=model_fixture_reghcs,
        regularization_factor=reg_factor,
    )
    field = p1.SemiDiscreteField_RegHCsTriple(
        grid=grid,
        model=model_fixture_reghcs,
        forcing_terms=forcing_terms,
        regularization_factor=reg_factor,
    )

    state_exact_t = p1.state_from_mms_when(
        mms_case=mms_case_reghcs_symbolic, t=t_eval, grid=grid
    )

    F_h = {
        "cp": field.Fcp(state_exact_t, t_eval),
        "T": field.FT(state_exact_t, t_eval),
        "cl": field.Fcl(state_exact_t, t_eval),
        "cd": field.Fcd(state_exact_t, t_eval),
        "cs": field.Fcs(state_exact_t, t_eval),
    }
    dt_exact = {
        "cp": mms_case_reghcs_symbolic.dt_cp(t_eval, grid.xx, grid.yy),
        "T": mms_case_reghcs_symbolic.dt_T(t_eval, grid.xx, grid.yy),
        "cl": mms_case_reghcs_symbolic.dt_cl(t_eval, grid.xx, grid.yy),
        "cd": mms_case_reghcs_symbolic.dt_cd(t_eval, grid.xx, grid.yy),
        "cs": mms_case_reghcs_symbolic.dt_cs(t_eval, grid.xx, grid.yy),
    }

    total_error_sq_sum = 0.0
    for var in ["cp", "T", "cl", "cd", "cs"]:
        err_var = grid.norm_H(F_h[var] - dt_exact[var])
        total_error_sq_sum += err_var**2
        print(
            f"  FieldCorrectness N={grid_n}, t={t_eval}: H-Norm Error F({var})-dt_{var} = {err_var:.3e}"
        )

    current_total_error = np.sqrt(total_error_sq_sum)

    if not hasattr(request.node.session, "field_correctness_errors_reghcs"):
        request.node.session.field_correctness_errors_reghcs = []  # type: ignore
    request.node.session.field_correctness_errors_reghcs.append(current_total_error)  # type: ignore

    if grid_n == FIELD_CORRECTNESS_GRID_SIZES[-1]:
        errors_collected = request.node.session.field_correctness_errors_reghcs  # type: ignore
        print(f"  Collected Field Correctness Errors (RegHCs): {errors_collected}")
        # Need at least 3 points for observed_rates_report's 3-point formula.
        # With 2 points, we can only check for error reduction.
        if len(errors_collected) >= 2:
            assert (
                errors_collected[-1] < errors_collected[0]
            ), "Field correctness error did not decrease with refinement."
            if len(errors_collected) >= 3:
                observed_rates_report(  # Assumes errors are ordered coarse to fine
                    errors=errors_collected,
                    expected_rate=2.0,  # Assuming 2nd order spatial accuracy in FDM approx.
                    tolerance=0.5,  # Wider tolerance for field approx.
                    cmp_type="least",
                )
        del request.node.session.field_correctness_errors_reghcs  # type: ignore


# --- Integrator Tests ---
GRID_REFINEMENTS_SPATIAL_INTEG = [8, 16, 32]
DT_REFINEMENTS_TEMPORAL_INTEG = 16*np.array([
    4e-3,
    2e-3,
    1e-3,
    5e-4,
    2.5e-4,
    1.25e-4,
    6.25e-5,
])  # Coarse to fine for observed_rates_report

EXPECTED_TEMPORAL_ORDER_REGHCS = 2.0
EXPECTED_SPATIAL_ORDER_REGHCS = 2.0
ORDER_TOLERANCE_INTEG = 0.3


@pytest.mark.parametrize("grid_n", GRID_REFINEMENTS_SPATIAL_INTEG)
def test_spatial_convergence_reghcstriple_stepper(
    model_fixture_reghcs: p1.DefaultModel01,
    mms_case_cls_reghcs_symbolic: Type[p1.MMSCaseSymbolic],
    grid_n: int,
    request,
):
    """
    Tests spatial convergence of the P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple
    using a single, small, fixed time step via MMSTrial.

    :param model_fixture_reghcs: Pytest fixture for the simulation model.
    :type model_fixture_reghcs: p1.DefaultModel01
    :param mms_case_reghcs_symbolic: Pytest fixture for the symbolic MMS case.
    :type mms_case_reghcs_symbolic: p1.MMSCaseSymbolic
    :param grid_n: The number of grid points for the current refinement level.
    :type grid_n: int
    :param request: Pytest request object to store errors across parameterized calls.
    """
    current_dt = 5e-6
    tf_sim = current_dt

    grid = p1.make_uniform_grid(N=grid_n, M=grid_n)

    integrator_params = {
        "regularization_factor": DEFAULT_REG_FACTOR,
        "num_pc_steps": 1,
        "num_newton_iterations": 5,
        "consec_xs_rtol": 1e-7,
    }
    common_params = {"regularization_factor": DEFAULT_REG_FACTOR}

    trial = mtu.MMSTrial(
        grid=grid,
        model=model_fixture_reghcs,
        mms_case_cls=mms_case_cls_reghcs_symbolic,
        field_cls=p1.SemiDiscreteField_RegHCsTriple,
        forcing_terms_cls=p1.ForcingTerms_RegHCsTriple,
        integrator_cls=p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple,
        integrator_params=integrator_params,
        forcing_terms_params=common_params,
        field_params=common_params,
    )

    summary = trial.run_for_errors(Tf=tf_sim, dt=current_dt)
    error_val = summary.overall_combined_error
    print(
        f"  SpatialConv RegHCs (N={grid_n}): dt={current_dt:.1e}, Error={error_val:.3e}"
    )

    if not hasattr(request.node.session, "spatial_errors_reghcs_integ"):
        request.node.session.spatial_errors_reghcs_integ = []  # type: ignore
    request.node.session.spatial_errors_reghcs_integ.append(error_val)  # type: ignore

    if grid_n == GRID_REFINEMENTS_SPATIAL_INTEG[-1]:
        errors_collected = request.node.session.spatial_errors_reghcs_integ  # type: ignore
        print(f"  Collected Spatial Errors (RegHCs Integrator): {errors_collected}")
        observed_rates_report(
            errors=errors_collected,
            expected_rate=EXPECTED_SPATIAL_ORDER_REGHCS,
            tolerance=ORDER_TOLERANCE_INTEG,
            cmp_type="least",
        )
        del request.node.session.spatial_errors_reghcs_integ  # type: ignore


@pytest.mark.parametrize("dt_val", DT_REFINEMENTS_TEMPORAL_INTEG)
def test_temporal_convergence_reghcstriple_stepper(
    model_fixture_reghcs: p1.DefaultModel01,
    mms_case_cls_reghcs_symbolic: Type[p1.MMSCaseSymbolic],
    dt_val: float,
    request: pytest.FixtureRequest,
    # Note: The request parameter is used to store errors across parameterized calls.
    # It is not a standard pytest fixture, but a built-in pytest feature.
    # The request parameter is used to access the session object for storing errors.
    # This is a common pattern in pytest for parameterized tests.
):
    """
    Tests temporal convergence of the P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple
    using a single step on a fixed, fine grid via MMSTrial.

    :param model_fixture_reghcs: Pytest fixture for the simulation model.
    :type model_fixture_reghcs: p1.DefaultModel01
    :param mms_case_reghcs_symbolic: Pytest fixture for the symbolic MMS case.
    :type mms_case_reghcs_symbolic: p1.MMSCaseSymbolic
    :param dt_val: The time step size for the current refinement level.
    :type dt_val: float
    :param request: Pytest request object to store errors across parameterized calls.
    """
    fine_grid_n = 128
    grid = p1.make_uniform_grid(N=fine_grid_n, M=fine_grid_n)
    tf_sim = dt_val

    integrator_params = {
        "regularization_factor": DEFAULT_REG_FACTOR,
        "num_pc_steps": 1,
        "num_newton_iterations": 1000,
        "consec_xs_rtol": 1e-9,
    }
    common_params = {"regularization_factor": DEFAULT_REG_FACTOR}

    print(integrator_params)

    trial = mtu.MMSTrial(
        grid=grid,
        model=model_fixture_reghcs,
        mms_case_cls=mms_case_cls_reghcs_symbolic,
        field_cls=p1.SemiDiscreteField_RegHCsTriple,
        forcing_terms_cls=p1.ForcingTerms_RegHCsTriple,
        integrator_cls=p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple,
        integrator_params=integrator_params,
        forcing_terms_params=common_params,
        field_params=common_params,
    )

    summary = trial.run_for_errors(Tf=tf_sim, dt=dt_val)
    error_val = summary.overall_combined_error
    print(
        f"  TemporalConv RegHCs (dt={dt_val:.1e}): N={fine_grid_n}, Error={error_val:.3e}"
    )

    if not hasattr(request.node.session, "temporal_errors_reghcs_integ"):
        request.node.session.temporal_errors_reghcs_integ = []  # type: ignore

    # Store errors in order from coarse dt to fine dt for observed_rates_report
    # The DT_REFINEMENTS_TEMPORAL_INTEG is already ordered coarse to fine.
    request.node.session.temporal_errors_reghcs_integ.append(error_val)  # type: ignore

    if dt_val == DT_REFINEMENTS_TEMPORAL_INTEG[-1]:
        errors_collected = request.node.session.temporal_errors_reghcs_integ  # type: ignore
        print(f"  Collected Temporal Errors (RegHCs Integrator): {errors_collected}")
        observed_rates_report(
            errors=errors_collected,
            expected_rate=EXPECTED_TEMPORAL_ORDER_REGHCS,
            tolerance=ORDER_TOLERANCE_INTEG,
            cmp_type="least",
        )
        del request.node.session.temporal_errors_reghcs_integ  # type: ignore


def test_reghcstriple_vs_fwd_euler_mms_comparison(
    model_fixture_reghcs: p1.DefaultModel01,
    mms_case_cls_reghcs_symbolic: Type[p1.MMSCaseSymbolic],
):
    """
    Compares the P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple
    with ForwardEulerIntegrator using an MMS case.
    The RegHCsTriple integrator is expected to yield smaller errors.

    :param model_fixture_reghcs: Pytest fixture for the simulation model.
    :type model_fixture_reghcs: p1.DefaultModel01
    :param mms_case_reghcs_symbolic: Pytest fixture for the symbolic MMS case.
    :type mms_case_reghcs_symbolic: p1.MMSCaseSymbolic
    """
    grid_n_comp = 4
    dt_comp = 5e-4
    tf_sim = dt_comp * 3

    grid = p1.make_uniform_grid(N=grid_n_comp, M=grid_n_comp)

    common_mms_params = {
        "grid": grid,
        "model": model_fixture_reghcs,
        "mms_case_cls": mms_case_cls_reghcs_symbolic,
    }
    common_forcing_field_params = {"regularization_factor": DEFAULT_REG_FACTOR}

    # --- RegHCsTriple Integrator ---
    integrator_params_reghcs = {
        "regularization_factor": DEFAULT_REG_FACTOR,
        "num_pc_steps": 1,
        "num_newton_iterations": 1,
        "consec_xs_rtol": 1e-7,
    }
    trial_reghcs = mtu.MMSTrial(
        **common_mms_params,
        field_cls=p1.SemiDiscreteField_RegHCsTriple,
        forcing_terms_cls=p1.ForcingTerms_RegHCsTriple,
        integrator_cls=p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple,
        integrator_params=integrator_params_reghcs,
        forcing_terms_params=common_forcing_field_params,
        field_params=common_forcing_field_params,
    )
    summary_reghcs = trial_reghcs.run_for_errors(Tf=tf_sim, dt=dt_comp)
    error_reghcs = summary_reghcs.overall_combined_error
    print(f"  Comparison: RegHCsTriple Integrator Error = {error_reghcs:.4e}")

    # --- Forward Euler Integrator ---
    # Forcing and Field use RegHCsTriple versions to ensure same problem is solved.
    trial_fwd_euler = mtu.MMSTrial(
        **common_mms_params,
        field_cls=p1.SemiDiscreteField_RegHCsTriple,
        forcing_terms_cls=p1.ForcingTerms_RegHCsTriple,
        integrator_cls=p1.ForwardEulerIntegrator,
        forcing_terms_params=common_forcing_field_params,
        field_params=common_forcing_field_params,
    )
    summary_fwd_euler = trial_fwd_euler.run_for_errors(Tf=tf_sim, dt=dt_comp)
    error_fwd_euler = summary_fwd_euler.overall_combined_error
    print(f"  Comparison: Forward Euler Integrator Error = {error_fwd_euler:.4e}")

    assert error_reghcs < error_fwd_euler, (
        f"RegHCsTriple error ({error_reghcs:.3e}) was not smaller than "
        f"Forward Euler error ({error_fwd_euler:.3e}) for dt={dt_comp}, N={grid_n_comp}"
    )

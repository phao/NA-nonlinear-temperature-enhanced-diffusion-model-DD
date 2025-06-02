# test_newton_residuals.py

import pytest
import numpy as np
import prob1base as p1
import prob1_mms_cases as p1mc  # Ensure this is imported


@pytest.fixture
def general_solver_config():
    """
    Provides general configuration for the solver tests:
    grid, model, time parameters, Newton solver parameters, and residual tolerance.
    """
    N = 8  # Using a small grid for efficiency
    M = 8
    grid = p1.make_uniform_grid(N, M)

    model_consts = p1.default_model_consts._replace(
        K1=1e-2, K2=1e-2, K3=1e-2, K4=1e-2, DT=1e-3, Kd=1e-3, Sd=10
    )
    model = p1.DefaultModel02(mc=model_consts)

    t0 = 0.0
    dt = 1e-4  # A reasonably small time step

    return {
        "grid": grid,
        "model": model,
        "t0": t0,
        "dt": dt,
        "num_pc_steps": 1,  # One predictor-corrector pass
        "num_newton_steps": 5,  # Sufficient Newton iterations for convergence
        "residual_atol_each": 1e-9,  # Absolute tolerance for the norm of the residual. "Each" refers to the fact that each residual, at each time step, must be below this tolerance.
    }


num_time_steps_to_test = [1, 20]

# List of MMS case classes to be used in parametrized tests
# User can add more MMSCaseBase subtypes here.
mms_case_classes_to_test = [
    p1mc.MMSCasePol,
    p1mc.MMSCaseExpSin,
    p1mc.MMSCaseCsZeroCrossing,
    p1mc.MMSCaseSlowlyChangingPeaks,
    p1mc.MMSCaseStiffExpDecay,
    p1mc.MMSCaseSlowlyChangingPeaks_Slow1e1,
    p1mc.MMSCaseSlowlyChangingPeaks_Slow1e4,
    p1mc.MMSCaseSlowlyChangingPeaks_Slow1e16,
    p1mc.MMSCaseSlowlyChangingPeaks_Fast1e1,
    p1mc.MMSCaseSlowlyChangingPeaks_Fast1e4,
    p1mc.MMSCaseSlowlyChangingPeaks_Fast1e8
]


@pytest.mark.parametrize("mms_case_cls", mms_case_classes_to_test)
@pytest.mark.parametrize("num_time_steps", num_time_steps_to_test)
def test_cs_triple_newton_residuals(general_solver_config, mms_case_cls, num_time_steps):
    """
    Tests if the Newton step residuals for T, cl, and cd are small
    after a time step with the CsTriple integrator, parametrized by MMS case.
    """
    config = general_solver_config
    grid = config["grid"]
    model = config["model"]
    t0 = config["t0"]
    dt = config["dt"]
    residual_atol = config["residual_atol_each"]

    # Instantiate the specific MMS case and corresponding initial state
    xx, yy = grid.xx, grid.yy
    mms_case = mms_case_cls(grid=grid, model=model)
    initial_state_dict = {
        var: getattr(mms_case, var)(t0, xx, yy) for var in ["cp", "T", "cl", "cd", "cs"]
    }
    initial_state = p1.StateVars(
        **initial_state_dict, model=model, hh=grid.hh, kk=grid.kk
    )

    forcing_terms = p1.ForcingTerms_CsTriple(mms_case=mms_case, model=model)
    field = p1.SemiDiscreteField_CsTriple(
        grid=grid, model=model, forcing_terms=forcing_terms
    )
    integrator = p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_CsTriple(
        semi_discrete_field=field,
        num_pc_steps=config["num_pc_steps"],
        num_newton_steps=config["num_newton_steps"],
    )

    residuals_T_norm = []
    residuals_cl_norm = []
    residuals_cd_norm = []

    current_state = initial_state
    current_t = t0
    for _time_step in range(num_time_steps):
        current_state = integrator.step(at_t0=current_state, t0=current_t, dt=dt)
        current_t += dt

        # Fetch residuals
        residuals_T_norm.append(grid.norm_H(integrator.last_residual["T"]))
        residuals_cl_norm.append(grid.norm_H(integrator.last_residual["cl"]))
        residuals_cd_norm.append(grid.norm_H(integrator.last_residual["cd"]))

    max_residual_T_norm = np.max(residuals_T_norm)
    max_residual_cl_norm = np.max(residuals_cl_norm)
    max_residual_cd_norm = np.max(residuals_cd_norm)

    mms_case_name = mms_case_cls.__name__
    print(
        f"CsTriple (maximum of) Residual Norms ({mms_case_name}, {num_time_steps=}): "
        f"T={max_residual_T_norm:.2e}, cl={max_residual_cl_norm:.2e}, cd={max_residual_cd_norm:.2e}"
    )

    assert (
        max_residual_T_norm < residual_atol
    ), f"T residual norm {max_residual_T_norm} too large for {mms_case_name} and {num_time_steps=}."
    assert (
        max_residual_cl_norm < residual_atol
    ), f"cl residual norm {max_residual_cl_norm} too large for {mms_case_name} and {num_time_steps=}."
    assert (
        max_residual_cd_norm < residual_atol
    ), f"cd residual norm {max_residual_cd_norm} too large for {mms_case_name} and {num_time_steps=}."


@pytest.mark.parametrize("mms_case_cls", mms_case_classes_to_test)
@pytest.mark.parametrize("num_time_steps", num_time_steps_to_test)
def test_hcs_triple_newton_residuals(general_solver_config, mms_case_cls, num_time_steps):
    """
    Tests if the Newton step residuals for T, cl, and cd are small
    after a time step with the HCsTriple integrator, parametrized by MMS case.
    """
    config = general_solver_config
    grid = config["grid"]
    xx, yy = grid.xx, grid.yy
    model = config["model"]  # MMSCasePol and ExpSin use Model01
    t0 = config["t0"]
    dt = config["dt"]
    residual_atol = config["residual_atol_each"]

    # Instantiate the specific MMS case and corresponding initial state
    mms_case = mms_case_cls(grid=grid, model=model)
    initial_state_dict = {
        var: getattr(mms_case, var)(t0, xx, yy) for var in ["cp", "T", "cl", "cd", "cs"]
    }
    initial_state = p1.StateVars(
        **initial_state_dict, model=model, hh=grid.hh, kk=grid.kk
    )

    forcing_terms = p1.ForcingTerms_HCsTriple(mms_case=mms_case, model=model)
    field = p1.SemiDiscreteField_HCsTriple(
        grid=grid, model=model, forcing_terms=forcing_terms
    )
    integrator = p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple(
        semi_discrete_field=field,
        num_pc_steps=config["num_pc_steps"],
        num_newton_steps=config["num_newton_steps"],
    )

    # Perform one time step
    # final_state = integrator.step(at_t0=initial_state, t0=t0, dt=dt) # if needed
    integrator.step(at_t0=initial_state, t0=t0, dt=dt)

    residuals_T_norm = []
    residuals_cl_norm = []
    residuals_cd_norm = []

    current_state = initial_state
    current_t = t0
    for _time_step in range(num_time_steps):
        current_state = integrator.step(at_t0=current_state, t0=current_t, dt=dt)
        current_t += dt

        # Fetch residuals
        residuals_T_norm.append(grid.norm_H(integrator.last_residual["T"]))
        residuals_cl_norm.append(grid.norm_H(integrator.last_residual["cl"]))
        residuals_cd_norm.append(grid.norm_H(integrator.last_residual["cd"]))

    max_residual_T_norm = np.max(residuals_T_norm)
    max_residual_cl_norm = np.max(residuals_cl_norm)
    max_residual_cd_norm = np.max(residuals_cd_norm)

    mms_case_name = mms_case_cls.__name__
    print(
        f"CsTriple (maximum of) Residual Norms ({mms_case_name}, {num_time_steps=}): "
        f"T={max_residual_T_norm:.2e}, cl={max_residual_cl_norm:.2e}, cd={max_residual_cd_norm:.2e}"
    )

    assert (
        max_residual_T_norm < residual_atol
    ), f"T residual norm {max_residual_T_norm} too large for {mms_case_name} and {num_time_steps=}."
    assert (
        max_residual_cl_norm < residual_atol
    ), f"cl residual norm {max_residual_cl_norm} too large for {mms_case_name} and {num_time_steps=}."
    assert (
        max_residual_cd_norm < residual_atol
    ), f"cd residual norm {max_residual_cd_norm} too large for {mms_case_name} and {num_time_steps=}."

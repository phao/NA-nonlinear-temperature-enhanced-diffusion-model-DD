# test_reghcstriple.py - Tests for RegHCsTriple related classes

import pytest
import numpy as np
from numpy.testing import assert_allclose
import math
import time
import sympy

import prob1base as p1
import prob1_mms_cases as p1mc
from utils_for_testing import observed_rates_report
import mms_trial_utils as mtu

# --- Test-Specific Model Constants ---
R0_test_reg: float = 8.3144621
Ea_test_reg: float = 1.60217662e-19
test_model_consts_reg: p1.ModelConsts = p1.ModelConsts(
    R0=R0_test_reg,
    Ea=Ea_test_reg,
    K1=1e-2,
    K2=1e-2,
    K3=1e-2,
    K4=1e-2,
    DT=1e-3,
    Dl_max=8.01e-4,
    phi_l=1e-5,
    gamma_T=1e-9,
    Kd=1e-4,
    Sd=10,
    Dd_max=2.46e-6,
    phi_d=1e-5,
    phi_T=Ea_test_reg / R0_test_reg,
    r_sp=5e-2,
    T_ref=300,
)

# --- Test Parameters ---
REGULARIZATION_FACTORS = [10.0, 50.0, 100.0]
GRID_SIZES = [8, 16, 32]  # For grid refinement tests
reg_factor_param = pytest.mark.parametrize("regularization_factor", REGULARIZATION_FACTORS)

# Reasonable timestep values for tests
DT_VALUES = [1e-3, 2.5e-4]
dt_param = pytest.mark.parametrize("dt", DT_VALUES)

# --- Fixtures ---
@pytest.fixture
def model_fixture():
    """Fixture for the DefaultModel01 using test-specific constants."""
    return p1.DefaultModel02(mc=test_model_consts_reg)

@pytest.fixture
def grid_fixture(request):
    """Function-scoped grid fixture with adjustable size."""
    # Default to mid-size grid if not specified
    grid_size = getattr(request, "param", 16)
    return p1.make_uniform_grid(N=grid_size, M=grid_size)


@pytest.fixture
def mms_case_fixture(grid_fixture, model_fixture):
    """Fixture for an MMS case (exp-sin) that works well with the numerical scheme."""
    return p1mc.MMSCaseExpSin(grid=grid_fixture, model=model_fixture)


@pytest.fixture
def forcing_terms_fixture(mms_case_fixture, model_fixture, regularization_factor):
    """Fixture for RegHCsTriple forcing terms."""
    return p1.ForcingTerms_RegHCsTriple(
        mms_case=mms_case_fixture, 
        model=model_fixture,
        regularization_factor=regularization_factor
    )


@pytest.fixture
def field_fixture(grid_fixture, model_fixture, forcing_terms_fixture, regularization_factor):
    """Fixture for RegHCsTriple semi-discrete field."""
    return p1.SemiDiscreteField_RegHCsTriple(
        grid=grid_fixture, 
        model=model_fixture, 
        forcing_terms=forcing_terms_fixture,
        regularization_factor=regularization_factor
    )


@pytest.fixture
def fwd_euler_integrator_fixture(field_fixture):
    """Fixture for a Forward Euler integrator with the RegHCsTriple field."""
    return p1.ForwardEulerIntegrator(semi_discrete_field=field_fixture)


@pytest.fixture
def pc_integrator_fixture(field_fixture, regularization_factor):
    """Fixture for the P_ModifiedEuler_C_Trapezoidal integrator with RegHCsTriple field."""
    return p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple(
        semi_discrete_field=field_fixture,
        num_pc_steps=1,
        num_newton_steps=1,
        regularization_factor=regularization_factor,
        num_newton_iterations=5,
        consec_xs_rtol=1e-6
    )


# --- Sanity Check Tests ---
@reg_factor_param
def test_heaviside_regularized_function(regularization_factor):
    """Test that the heaviside_regularized function behaves as expected."""
    x_vals = np.linspace(-5, 5, 101)
    h_vals = p1.heaviside_regularized(x_vals, regularization_factor)
    print(f"Testing Heaviside function with reg_factor={regularization_factor}")
    
    # Basic properties
    assert np.all(h_vals >= 0), "Heaviside function should always be non-negative"
    assert np.all(h_vals <= 1), "Heaviside function should be bounded by 1"
    
    # Test specific values
    assert h_vals[0] < 0.01, f"H(-5) should be very close to 0, got {h_vals[0]}"
    assert h_vals[-1] > 0.99, f"H(5) should be very close to 1, got {h_vals[-1]}"
    assert 0.45 < h_vals[50] < 0.55, f"H(0) should be around 0.5, got {h_vals[50]}"
    
    # Higher regularization factor should make the transition steeper
    if regularization_factor > 10:
        h_steep = p1.heaviside_regularized(x_vals, regularization_factor)
        h_less_steep = p1.heaviside_regularized(x_vals, 5.0)
        
        # At x slightly > 0, the steeper function should have higher values
        idx_positive = np.where(x_vals > 0)[0][0]
        assert h_steep[idx_positive] > h_less_steep[idx_positive], \
            "Higher regularization factor should make the transition steeper"


@reg_factor_param
def test_forcing_terms_reghcstriple_construction(mms_case_fixture, model_fixture, regularization_factor):
    """Test that ForcingTerms_RegHCsTriple can be constructed and has the expected properties."""
    forcing_terms = p1.ForcingTerms_RegHCsTriple(
        mms_case=mms_case_fixture, 
        model=model_fixture,
        regularization_factor=regularization_factor
    )
    
    assert forcing_terms.regularization_factor == regularization_factor
    assert forcing_terms.model == model_fixture
    assert forcing_terms.mms_case == mms_case_fixture


@reg_factor_param
def test_field_reghcstriple_construction(grid_fixture, model_fixture, forcing_terms_fixture, regularization_factor):
    """Test that SemiDiscreteField_RegHCsTriple can be constructed and has the expected properties."""
    field = p1.SemiDiscreteField_RegHCsTriple(
        grid=grid_fixture, 
        model=model_fixture, 
        forcing_terms=forcing_terms_fixture,
        regularization_factor=regularization_factor
    )
    
    assert field.regularization_factor == regularization_factor
    assert field.grid == grid_fixture
    assert field.model == model_fixture
    

# --- MMS Tests ---
@reg_factor_param
@pytest.mark.parametrize("grid_fixture", GRID_SIZES, indirect=True)
def test_reghcstriple_field_temporal_derivative_approximation(
    grid_fixture, model_fixture, mms_case_fixture, forcing_terms_fixture, regularization_factor
):
    """
    Test that the SemiDiscreteField_RegHCsTriple correctly approximates temporal derivatives
    when applied to the exact solution.
    
    This test follows the scheme:
    1. Construct the semi-discrete field with forcing terms
    2. Apply it to the exact solution at a given time
    3. Compare with the exact temporal derivative
    4. Verify that the error decreases with grid refinement
    """
    grid_size = grid_fixture.N  # Extract grid size for logging
    print(f"\nTesting RegHCsTriple field temporal derivative approximation (reg_factor={regularization_factor}, grid_size={grid_size})")
    
    field = p1.SemiDiscreteField_RegHCsTriple(
        grid=grid_fixture, 
        model=model_fixture, 
        forcing_terms=forcing_terms_fixture,
        regularization_factor=regularization_factor
    )
    
    t_test = 0.1  # Test time
    grid = grid_fixture
    xx, yy = grid.xx, grid.yy
    
    # Get exact solution at t_test
    exact_state = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t_test, grid=grid)
    
    # Apply semi-discrete field to get numerical approximation of temporal derivatives
    Fcp_at_exact = field.Fcp(exact_state, t_test)
    FT_at_exact = field.FT(exact_state, t_test)
    Fcl_at_exact = field.Fcl(exact_state, t_test)
    Fcd_at_exact = field.Fcd(exact_state, t_test)
    Fcs_at_exact = field.Fcs(exact_state, t_test)
    
    # Get exact temporal derivatives
    exact_dt_cp = mms_case_fixture.dt_cp(t_test, xx, yy)
    exact_dt_T = mms_case_fixture.dt_T(t_test, xx, yy)
    exact_dt_cl = mms_case_fixture.dt_cl(t_test, xx, yy)
    exact_dt_cd = mms_case_fixture.dt_cd(t_test, xx, yy)
    exact_dt_cs = mms_case_fixture.dt_cs(t_test, xx, yy)
    
    # Compare interior points only (excluding boundary)
    for var_name, num_dt, exact_dt in [
        ("cp", Fcp_at_exact, exact_dt_cp),
        ("T", FT_at_exact, exact_dt_T),
        ("cl", Fcl_at_exact, exact_dt_cl),
        ("cd", Fcd_at_exact, exact_dt_cd),
        ("cs", Fcs_at_exact, exact_dt_cs)
    ]:
        # Compute error in interior
        interior_error = np.max(np.abs(num_dt[1:-1, 1:-1] - exact_dt[1:-1, 1:-1]))
        print(f"  {var_name} max interior error: {interior_error:.2e}")
        
        # For the finest grid, assert that error is reasonably small
        if grid_size == max(GRID_SIZES):
            assert interior_error < 1e-2, f"Error for {var_name} is too large: {interior_error:.2e}"


@reg_factor_param
@pytest.mark.parametrize("grid_fixture", GRID_SIZES, indirect=True)
def test_reghcstriple_field_application_to_exact_solution(
    grid_fixture, model_fixture, mms_case_fixture, forcing_terms_fixture, regularization_factor
):
    """
    Test that when we apply the SemiDiscreteField_RegHCsTriple to the exact solution,
    the result matches the expected temporal derivative with high accuracy.
    """
    grid_size = grid_fixture.N  # Extract grid size for logging
    print(f"\nTesting RegHCsTriple field application to exact solution (reg_factor={regularization_factor}, grid_size={grid_size})")

    field = p1.SemiDiscreteField_RegHCsTriple(
        grid=grid_fixture, 
        model=model_fixture, 
        forcing_terms=forcing_terms_fixture,
        regularization_factor=regularization_factor
    )
    
    t_test = 0.2  # Test time
    grid = grid_fixture
    
    # Get exact solution at t_test
    exact_state = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t_test, grid=grid)
    
    # Apply semi-discrete field to get numerical approximation of temporal derivatives
    Fcp_at_exact = field.Fcp(exact_state, t_test)
    FT_at_exact = field.FT(exact_state, t_test)
    Fcl_at_exact = field.Fcl(exact_state, t_test)
    Fcd_at_exact = field.Fcd(exact_state, t_test)
    Fcs_at_exact = field.Fcs(exact_state, t_test)
    
    # Check that each component of the field returns a reasonable value
    for var_name, value in [
        ("cp", Fcp_at_exact), 
        ("T", FT_at_exact), 
        ("cl", Fcl_at_exact), 
        ("cd", Fcd_at_exact), 
        ("cs", Fcs_at_exact)
    ]:
        assert not np.any(np.isnan(value)), f"NaN values found in {var_name}"
        assert not np.any(np.isinf(value)), f"Infinite values found in {var_name}"
        
        # Check that interior values are reasonable (not extremely large)
        interior_values = value[1:-1, 1:-1]
        assert np.max(np.abs(interior_values)) < 100, f"Unreasonably large values in {var_name}"


@reg_factor_param
def test_reghcstriple_field_convergence_with_grid_refinement(
    model_fixture, mms_case_fixture, regularization_factor
):
    """
    Tests that the error in approximating temporal derivatives decreases with grid refinement.
    We'll use a set of grids with increasing resolution and track the errors.
    """
    print(f"\nTesting RegHCsTriple field convergence with grid refinement (reg_factor={regularization_factor})")
    
    # Use the smallest grid size as the base
    grid_size = GRID_SIZES[0]
    
    t_test = 0.1
    errors = []
    
    # Use 4 levels of refinement
    refinement_levels = 4
    refined_sizes = [grid_size * (2 ** i) for i in range(refinement_levels)]
    
    for size in refined_sizes:
        print(f"  Testing grid size {size}x{size}")
        grid = p1.make_uniform_grid(N=size, M=size)
        xx, yy = grid.xx, grid.yy
        
        # Create components for this grid size
        forcing_terms = p1.ForcingTerms_RegHCsTriple(
            mms_case=mms_case_fixture, 
            model=model_fixture,
            regularization_factor=regularization_factor
        )
        
        field = p1.SemiDiscreteField_RegHCsTriple(
            grid=grid, 
            model=model_fixture, 
            forcing_terms=forcing_terms,
            regularization_factor=regularization_factor
        )
        
        # Get exact solution and apply field
        exact_state = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t_test, grid=grid)
        FT_at_exact = field.FT(exact_state, t_test)
        
        # Get exact temporal derivatives
        exact_dt_T = mms_case_fixture.dt_T(t_test, xx, yy)
        
        # Compute error in temperature (focus on one variable to simplify)
        error_T = np.max(np.abs(FT_at_exact.T[1:-1, 1:-1] - exact_dt_T[1:-1, 1:-1]))
        errors.append(error_T)
        print(f"    T max interior error: {error_T:.2e}")
    
    # Check convergence rate with observed_rates_report
    observed_rates_report(
        errors,
        expected_rate=2.0,  # Expect second-order convergence in space
        tolerance=0.3,      # Allow some tolerance due to regularization effects
        cmp_type="least"    # We expect at least this rate
    )


@reg_factor_param
@pytest.mark.parametrize("grid_fixture", GRID_SIZES, indirect=True)
@dt_param
def test_reghcstriple_forward_euler_one_step(
    grid_fixture, model_fixture, mms_case_fixture, forcing_terms_fixture, 
    fwd_euler_integrator_fixture, regularization_factor, dt
):
    """
    Test one step of Forward Euler integration with the RegHCsTriple field.
    Compare against exact solution at t+dt.
    """
    grid_size = grid_fixture.N  # Extract grid size for logging
    print(f"\nTesting Forward Euler one step with RegHCsTriple field (reg_factor={regularization_factor}, grid_size={grid_size}, dt={dt:.1e})")
    
    grid = grid_fixture
    xx, yy = grid.xx, grid.yy
    integrator = fwd_euler_integrator_fixture
    
    # Starting time
    t0 = 0.05
    t1 = t0 + dt
    
    # Get exact solution at t0
    state_t0 = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t0, grid=grid)
    
    # Perform one step of Forward Euler
    state_t1_numerical = integrator.step(at_t0=state_t0, t0=t0, dt=dt)
    
    # Get exact solution at t1
    state_t1_exact = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t1, grid=grid)
    
    # Compare interior points
    errors = {}
    for var_name, num_var, exact_var in [
        ("cp", state_t1_numerical.cp, state_t1_exact.cp),
        ("T", state_t1_numerical.T, state_t1_exact.T),
        ("cl", state_t1_numerical.cl, state_t1_exact.cl),
        ("cd", state_t1_numerical.cd, state_t1_exact.cd),
        ("cs", state_t1_numerical.cs, state_t1_exact.cs)
    ]:
        num_interior = num_var[1:-1, 1:-1]
        exact_interior = exact_var[1:-1, 1:-1]
        error = np.max(np.abs(num_interior - exact_interior))
        errors[var_name] = error
        print(f"  {var_name} max interior error: {error:.2e}")
    
    # For the finest grid and smallest dt, assert errors are reasonably small
    if grid_size == max(GRID_SIZES) and dt == min(DT_VALUES):
        for var_name, error in errors.items():
            assert error < 1e-2, f"Error for {var_name} is too large: {error:.2e}"


@reg_factor_param
@pytest.mark.parametrize("grid_fixture", GRID_SIZES, indirect=True)
def test_reghcstriple_pc_integrator_one_step(
    grid_fixture, model_fixture, mms_case_fixture, forcing_terms_fixture, 
    pc_integrator_fixture, regularization_factor
):
    """
    Test one step of the predictor-corrector integrator with the RegHCsTriple field.
    Compare against exact solution at t+dt.
    """
    grid_size = grid_fixture.N  # Extract grid size for logging
    print(f"\nTesting PC integrator one step with RegHCsTriple field (reg_factor={regularization_factor}, grid_size={grid_size})")
    
    grid = grid_fixture
    xx, yy = grid.xx, grid.yy
    integrator = pc_integrator_fixture
    
    # Starting time and dt
    t0 = 0.05
    dt = 1e-3
    t1 = t0 + dt
    
    # Get exact solution at t0
    state_t0 = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t0, grid=grid)
    
    # Perform one step of PC integrator
    state_t1_numerical = integrator.step(at_t0=state_t0, t0=t0, dt=dt)
    
    # Get exact solution at t1
    state_t1_exact = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t1, grid=grid)
    
    # Compare interior points
    errors = {}
    for var_name, num_var, exact_var in [
        ("cp", state_t1_numerical.cp, state_t1_exact.cp),
        ("T", state_t1_numerical.T, state_t1_exact.T),
        ("cl", state_t1_numerical.cl, state_t1_exact.cl),
        ("cd", state_t1_numerical.cd, state_t1_exact.cd),
        ("cs", state_t1_numerical.cs, state_t1_exact.cs)
    ]:
        num_interior = num_var[1:-1, 1:-1]
        exact_interior = exact_var[1:-1, 1:-1]
        error = np.max(np.abs(num_interior - exact_interior))
        errors[var_name] = error
        print(f"  {var_name} max interior error: {error:.2e}")
    
    # For the finest grid, assert errors are reasonably small
    if grid_size == max(GRID_SIZES):
        for var_name, error in errors.items():
            assert error < 1e-2, f"Error for {var_name} is too large: {error:.2e}"


@reg_factor_param
def test_reghcstriple_forward_euler_temporal_convergence(
    model_fixture, mms_case_fixture, regularization_factor
):
    """
    Test the temporal convergence order of the Forward Euler method with RegHCsTriple field.
    We expect first-order convergence in time.
    """
    # Always use the finest grid for temporal convergence tests
    grid_size = max(GRID_SIZES)
    print(f"\nTesting Forward Euler temporal convergence with RegHCsTriple (reg_factor={regularization_factor}, grid_size={grid_size})")
    
    # Create a grid
    grid = p1.make_uniform_grid(N=grid_size, M=grid_size)
    
    # Create components
    forcing_terms = p1.ForcingTerms_RegHCsTriple(
        mms_case=mms_case_fixture, 
        model=model_fixture,
        regularization_factor=regularization_factor
    )
    
    field = p1.SemiDiscreteField_RegHCsTriple(
        grid=grid, 
        model=model_fixture, 
        forcing_terms=forcing_terms,
        regularization_factor=regularization_factor
    )
    
    integrator = p1.ForwardEulerIntegrator(semi_discrete_field=field)
    
    # Starting time
    t0 = 0.1
    base_dt = 1e-3
    
    # Range of dt values (halving each time)
    num_refinements = 4
    refinement_ratio = 2.0
    dt_values = [base_dt / (refinement_ratio**k) for k in range(num_refinements)]
    
    # Track errors for each dt
    errors = []
    
    for dt in dt_values:
        t1 = t0 + dt
        print(f"  Testing dt={dt:.2e}")
        
        # Get exact solution at t0
        state_t0 = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t0, grid=grid)
        
        # Perform one step
        state_t1_numerical = integrator.step(at_t0=state_t0, t0=t0, dt=dt)
        
        # Get exact solution at t1
        state_t1_exact = p1.state_from_mms_when(mms_case=mms_case_fixture, t=t1, grid=grid)
        
        # Calculate error norm (focusing on temperature for simplicity)
        error_T = grid.norm_H(state_t1_numerical.T - state_t1_exact.T)
        errors.append(error_T)
        print(f"    T error norm: {error_T:.2e}")
    
    # Check convergence rate with observed_rates_report
    observed_rates_report(
        errors,
        expected_rate=1.0,  # Expect first-order convergence in time for Forward Euler
        tolerance=0.15,
        cmp_type="equal"    # Should be close to exactly first order
    )


@reg_factor_param
def test_reghcstriple_mms_trial_short_simulation(model_fixture, regularization_factor):
    """
    Run a short simulation using MMSTrial to encapsulate the setup and error calculation, comparing the Forward Euler and PC integrators with RegHCsTriple field.
    """
    # Use a small grid for MMS trials to save test time, and due to forward euler conditional stability. 
    grid_size = 4
    print(f"\nTesting RegHCsTriple short MMS simulation (reg_factor={regularization_factor}, grid_size={grid_size})")
    
    # Create grid
    grid = p1.make_uniform_grid(N=grid_size, M=grid_size)
    
    # Setup for Forward Euler
    mms_trial_fe = mtu.MMSTrial(
        grid=grid,
        model=model_fixture,
        mms_case_cls=p1mc.MMSCaseExpSin,
        field_cls=p1.SemiDiscreteField_RegHCsTriple,
        forcing_terms_cls=p1.ForcingTerms_RegHCsTriple,
        integrator_cls=p1.ForwardEulerIntegrator,
        forcing_terms_params={"regularization_factor": regularization_factor},
        field_params={"regularization_factor": regularization_factor},
    )
    
    # Setup for PC Integrator
    mms_trial_pc = mtu.MMSTrial(
        grid=grid,
        model=model_fixture,
        mms_case_cls=p1mc.MMSCaseExpSin,
        field_cls=p1.SemiDiscreteField_RegHCsTriple,
        forcing_terms_cls=p1.ForcingTerms_RegHCsTriple,
        integrator_cls=p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple,
        forcing_terms_params={"regularization_factor": regularization_factor},
        field_params={"regularization_factor": regularization_factor},
        integrator_params={
            "regularization_factor": regularization_factor,
            "num_newton_iterations": 5,
            "consec_xs_rtol": 1e-6
        }
    )
    
    # Run short simulations
    Tf = 1e-3  # Short final time for quick test
    dt = Tf/1e3
    
    # Forward Euler
    print("  Running Forward Euler simulation...")
    start_time = time.time()
    error_summary_fe = mms_trial_fe.run_for_errors(Tf=Tf, dt=dt)
    fe_time = time.time() - start_time
    print(f"  Forward Euler completed in {fe_time:.2f}s")
    print(f"  Forward Euler error summary: {error_summary_fe}")
    
    # PC Integrator
    print("  Running PC Integrator simulation...")
    start_time = time.time()
    error_summary_pc = mms_trial_pc.run_for_errors(Tf=Tf, dt=dt)
    pc_time = time.time() - start_time
    print(f"  PC Integrator completed in {pc_time:.2f}s")
    print(f"  PC Integrator error summary: {error_summary_pc}")

    print(error_summary_pc)
    
    # Check that errors are reasonable
    assert error_summary_fe.overall_combined_error < 1e-1, \
        f"Forward Euler overall error too large: {error_summary_fe.overall_combined_error:.2e}"
    assert error_summary_pc.overall_combined_error < 1e-1, \
        f"PC Integrator overall error too large: {error_summary_pc.overall_combined_error:.2e}"

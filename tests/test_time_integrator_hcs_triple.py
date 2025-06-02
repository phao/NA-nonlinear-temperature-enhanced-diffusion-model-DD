# test_time_integrator_hcs_triple.py
import pytest
import numpy as np
from numpy.testing import assert_allclose

import prob1base as p1
from typing import Callable, Dict, Tuple, List, Any, Type

# --- Test-Specific Model Constants ---
R0_test_ti: float = 8.3144621
Ea_test_ti: float = 1.60217662e-19
test_model_consts_ti: p1.ModelConsts = p1.ModelConsts(
    R0=R0_test_ti,
    Ea=Ea_test_ti,
    K1=1.2e-2,
    K2=1.3e-2,
    K3=1.4e-2,
    K4=1.5e-2,
    DT=1.2e-3,
    Dl_max=8.2e-4,
    phi_l=1.2e-5,
    gamma_T=1.2e-9,
    Kd=1.2e-8,
    Sd=13,
    Dd_max=2.6e-6,
    phi_d=1.2e-5,
    phi_T=Ea_test_ti / R0_test_ti,
    r_sp=5.2e-2,
    T_ref=301,
)


# --- Fixtures ---
@pytest.fixture
def grid_fixture_ti() -> p1.Grid:
    return p1.make_uniform_grid(N=4, M=4)


@pytest.fixture
def model_fixture_ti() -> p1.DefaultModel01:
    """Returns a fresh model instance for each test to avoid state leakage from Kd modifications."""
    return p1.DefaultModel01(mc=test_model_consts_ti)


@pytest.fixture
def mock_field_factory_ti(
    grid_fixture_ti: p1.Grid, model_fixture_ti: p1.DefaultModel01
):
    """
    Factory to create a mock SemiDiscreteField_HCsTriple.
    Allows controlling Fcs(t0) and fcs_source(t1) for specific Y0' values.
    """

    def _factory(
        Fcs_at_t0_val: float = 0.0, fcs_source_at_t1_val: float = 0.0
    ) -> Type[p1.SemiDiscreteField_HCsTriple]:  # Returns the class type

        class MockForcing(p1.ForcingTermsBase):
            def __init__(
                self, grid: p1.Grid, fcs_t1_val: float, fcs_t0_for_Fcs_val: float
            ):
                self._grid = grid
                self._fcs_t1_val = fcs_t1_val
                # This mock structure assumes Fcs_at_t0_val in MockField will handle the full Fcs(t0)
                # So, fcs_source(t0) can be zero here.
                self._fcs_t0_for_Fcs_val = fcs_t0_for_Fcs_val

            def fcp(self, t: float, xx, yy) -> np.ndarray:
                return self._grid.make_full0()

            def fT(self, t: float, xx, yy) -> np.ndarray:
                return self._grid.make_full0()

            def fcl(self, t: float, xx, yy) -> np.ndarray:
                return self._grid.make_full0()

            def fcd(self, t: float, xx, yy) -> np.ndarray:
                return self._grid.make_full0()

            def fcs(self, t: float, xx, yy) -> np.ndarray:  # This is fcs_source
                # If t is t0 (e.g., 0.0), return the part of Fcs_at_t0_val that is fcs_source(t0)
                # If t is t1, return fcs_source_at_t1_val
                if np.isclose(t, 0.0):  # t0
                    # This is tricky because Fcs(t0) = fcs_source(t0) - Reaction(t0)
                    # For the test, we directly mock Fcs(t0).
                    # For fcs_source(t1), we provide fcs_source_at_t1_val.
                    # To make Y0' simple, fcs_source(t0) embedded in Fcs_at_t0_val is easier.
                    return np.full(
                        self._grid.full_shape, self._fcs_t0_for_Fcs_val, dtype=float
                    )
                else:  # t1
                    return np.full(self._grid.full_shape, self._fcs_t1_val, dtype=float)

        class MockField(p1.SemiDiscreteField_HCsTriple):
            def __init__(
                self,
                grid: p1.Grid,
                model: p1.DefaultModel01,
                forcing_terms_instance: p1.ForcingTermsBase,
            ):
                super().__init__(
                    grid=grid, model=model, forcing_terms=forcing_terms_instance
                )
                self._Fcs_at_t0_val = Fcs_at_t0_val
                # self._Fcs_at_t1_star_val is not needed for corrector test's Y0 setup

            def Fcs(self, at_t: p1.StateVars, t: float) -> np.ndarray:
                if np.isclose(t, 0.0):  # t0 for Y0' calculation in corrector
                    return np.full(
                        self.grid.full_shape, self._Fcs_at_t0_val, dtype=float
                    )
                # For initial_cs_pred, it needs Fcs at t0 and t1_star
                # The mock_field_factory_ti now tailors inputs for specific methods
                # This simplified Fcs mock is for testing `corrector_cs_step`'s Y0 setup.
                # For `initial_cs_pred`, we need a different mocking strategy for Fcs.
                # Let's refine `mock_field_factory_ti` to handle this better.
                # For now, this structure is more for the corrector.
                # The `test_initial_cs_pred_no_clipping` will need its own mock field setup.
                raise NotImplementedError(
                    "MockField.Fcs needs specific values for t0 and t1_star for predictor."
                )

        # For corrector_cs_step, the fcs_source(t0) is implicitly part of Fcs_at_t0_val.
        mock_forcing = MockForcing(grid_fixture_ti, fcs_source_at_t1_val, 0.0)
        return MockField(grid_fixture_ti, model_fixture_ti, mock_forcing)  # type: ignore

    return _factory


@pytest.fixture
def state_vars_ti_factory(
    grid_fixture_ti: p1.Grid, model_fixture_ti: p1.DefaultModel01
):
    """Factory to create StateVars instances for time integrator tests."""

    def _factory(
        cs_val: float,
        cp_val: float = 0.1,
        T_val: float = 0.1,
        cl_val: float = 0.1,
        cd_val: float = 0.1,
    ) -> p1.StateVars:
        return p1.StateVars(
            cp=np.full(grid_fixture_ti.full_shape, cp_val, dtype=float),
            T=np.full(grid_fixture_ti.full_shape, T_val, dtype=float),
            cl=np.full(grid_fixture_ti.full_shape, cl_val, dtype=float),
            cd=np.full(grid_fixture_ti.full_shape, cd_val, dtype=float),
            cs=np.full(grid_fixture_ti.full_shape, cs_val, dtype=float),
            model=model_fixture_ti,
            hh=grid_fixture_ti.hh,
            kk=grid_fixture_ti.kk,
        )

    return _factory


@pytest.fixture
def integrator_hcs_ti(
    grid_fixture_ti: p1.Grid,
    model_fixture_ti: p1.DefaultModel01,  # Get a fresh model for the integrator
) -> p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple:
    """
    Fixture for the HCsTriple time integrator.
    Uses a default NoForcingTerms for its semi_discrete_field,
    as individual tests will often override semi_discrete_field.Fcs or provide specific fields.
    """
    no_forcing = p1.NoForcingTerms(grid_fixture_ti)
    # Create a real SemiDiscreteField_HCsTriple for the integrator's structure
    # Tests that need fine control over Fcs will swap this out or patch Fcs.
    field = p1.SemiDiscreteField_HCsTriple(
        grid=grid_fixture_ti, model=model_fixture_ti, forcing_terms=no_forcing
    )
    return p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple(
        semi_discrete_field=field, num_pc_steps=1, num_newton_steps=1
    )


# --- Tests for P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple ---


# --- initial_cs_pred ---
@pytest.mark.parametrize(
    "cs0_val, Fcs_t0_val_mock, Fcs_t1_star_val_mock, dt, expected_cs1_val_no_mask",
    [
        (1.0, 0.1, 0.2, 0.1, 1.0 + 0.5 * 0.1 * (0.1 + 0.2)),  # Expected: 1.015
        (0.01, -0.2, -0.2, 0.1, 0.01 + 0.5 * 0.1 * (-0.2 - 0.2)),  # Expected: -0.01
        (-0.1, -0.1, -0.1, 0.1, -0.1 + 0.5 * 0.1 * (-0.1 - 0.1)),  # Expected: -0.11
        (0.0, 0.0, 0.0, 0.1, 0.0),  # Expected: 0.0
    ],
)
def test_initial_cs_pred_no_clipping(
    integrator_hcs_ti: p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple,
    state_vars_ti_factory: Callable[..., p1.StateVars],
    grid_fixture_ti: p1.Grid,
    model_fixture_ti: p1.DefaultModel01,  # For creating specific mock field
    cs0_val: float,
    Fcs_t0_val_mock: float,
    Fcs_t1_star_val_mock: float,
    dt: float,
    expected_cs1_val_no_mask: float,
):
    """
    Tests initial_cs_pred: Cs1 = Cs0 + 0.5*dt*(Fcs(Cs0,t0) + Fcs(Cs_star,t1)).
    Verifies no internal positivity clipping, only boundary mask.
    Cs_star = Cs0 + dt*Fcs(Cs0,t0) (also no clipping here).
    """

    class TempMockField(p1.SemiDiscreteField_HCsTriple):
        def Fcs(self, at_t_state_vars: p1.StateVars, t: float) -> np.ndarray:
            # This mock needs to distinguish between the call for Fcs0 and Fcs_star
            # Fcs0 is called with at_t_state_vars.cs == cs0_val
            # Fcs_star is called with at_t_state_vars.cs being cs_star
            # For simplicity, we'll assume t tells us which call it is.
            if np.isclose(t, 0.0):  # t0
                return np.full(self.grid.full_shape, Fcs_t0_val_mock, dtype=float)
            else:  # t0 + dt (for Fcs_star)
                return np.full(self.grid.full_shape, Fcs_t1_star_val_mock, dtype=float)

    # Create a new field instance with the specific mock behavior for Fcs
    # Forcing terms can be NoForcingTerms as Fcs is overridden.
    specific_mock_field = TempMockField(
        grid=grid_fixture_ti,
        model=model_fixture_ti,
        forcing_terms=p1.NoForcingTerms(grid_fixture_ti),
    )
    integrator_hcs_ti.semi_discrete_field = specific_mock_field  # type: ignore

    at_t0_state: p1.StateVars = state_vars_ti_factory(cs_val=cs0_val)
    t0: float = 0.0

    cs1_predicted_masked: np.ndarray = integrator_hcs_ti.initial_cs_pred(
        at_t0_state, t0, dt=dt
    )

    assert_allclose(
        cs1_predicted_masked[1:-1, 1:-1],
        expected_cs1_val_no_mask,
        rtol=1e-9,
        atol=1e-12,
        err_msg="initial_cs_pred interior value mismatch",
    )
    # Check boundary points are zero
    assert np.all(cs1_predicted_masked[0, :] == 0.0)
    assert np.all(cs1_predicted_masked[-1, :] == 0.0)
    assert np.all(cs1_predicted_masked[:, 0] == 0.0)
    assert np.all(cs1_predicted_masked[:, -1] == 0.0)


# --- corrector_cs_step ---
# Formula for corrector:
# Y0_prime = 2*Cs0 + dt*( Fcs(Cs0,t0) + fcs_source(t1) )
# R1 = Kd*(Sd-Cd1)*(1+Cl1)
# if Y0_prime > tol: Cs1 = Y0_prime / (2 - dt*R1)
# if Y0_prime < -tol: Cs1 = Y0_prime / 2.0
# if |Y0_prime| <= tol: Cs1 = 0.0


@pytest.mark.parametrize(
    "cs0_val, Fcs_t0_mock_val, fcs_source_t1_mock_val, R1_target, dt, expected_cs1_val_no_mask",
    [
        # Case 1: Y0_prime > 0. del_Y1 = (2 - dt*R1_target) is positive.
        (
            0.1,
            0.5,
            0.5,
            0.5,
            0.1,
            (2 * 0.1 + 0.1 * (0.5 + 0.5)) / (2.0 - 0.1 * 0.5),
        ),  # Y0'=0.3, R1=0.5,delY1=1.95 -> 0.3/1.95=0.1538...
        (
            0.01,
            0.02,
            0.02,
            0.1,
            0.01,
            (2 * 0.01 + 0.01 * (0.02 + 0.02)) / (2.0 - 0.01 * 0.1),
        ),  # Y0'=0.0204,R1=0.1,delY1=1.999 -> 0.010205...
        # Case 2: Y0_prime < 0.
        (
            -0.1,
            -0.5,
            -0.5,
            0.5,
            0.1,
            (2 * (-0.1) + 0.1 * (-0.5 - 0.5)) / 2.0,
        ),  # Y0'=-0.3 -> -0.3/2 = -0.15
        (
            -0.01,
            -0.02,
            -0.02,
            10.0,
            0.1,
            (2 * (-0.01) + 0.1 * (-0.02 - 0.02)) / 2.0,
        ),  # Y0'=-0.024 -> -0.012 (R1 doesn't matter)
        # Case 3: Y0_prime is zero (or very close). Cs1 = 0.0
        (0.0, 0.0, 0.0, 0.5, 0.1, 0.0),  # Y0'=0
        (0.001, -0.02, 0.0, 1.0, 0.1, 0.0),  # Y0'=2*0.001+0.1*(-0.02+0)=0.002-0.002=0
        (0.0, 1e-12, -1e-12, 1.0, 0.1, 0.0),  # Y0' near zero
    ],
)
def test_corrector_cs_step_logic(
    integrator_hcs_ti: p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple,
    state_vars_ti_factory: Callable[..., p1.StateVars],
    model_fixture_ti: p1.DefaultModel01,  # Original model for Sd, and to restore Kd
    grid_fixture_ti: p1.Grid,
    cs0_val: float,
    Fcs_t0_mock_val: float,
    fcs_source_t1_mock_val: float,
    R1_target: float,
    dt: float,
    expected_cs1_val_no_mask: float,
):
    """Tests the algebraic logic of corrector_cs_step."""

    class TempMockFieldForCorrector(p1.SemiDiscreteField_HCsTriple):
        def Fcs(self, at_t_state: p1.StateVars, t: float) -> np.ndarray:
            # This is Fcs(Cs0, Cl0, Cd0, t0) = fcs_source(t0) - Reaction(Cs0,...)
            # For the corrector, we need Fcs(at_t0, t0).
            # We mock its value directly.
            if np.isclose(t, 0.0):  # t0
                return np.full(self.grid.full_shape, Fcs_t0_mock_val, dtype=float)
            raise ValueError(
                "MockField.Fcs called at unexpected time in corrector test."
            )

        # The fcs method on the field (self.fcs) points to the forcing term's fcs (fcs_source)
        # This will be handled by a MockForcing instance.

    class TempMockForcing(p1.ForcingTermsBase):
        def __init__(self, grid: p1.Grid, fcs_t1_val: float):
            self._grid = grid
            self._fcs_t1_val = fcs_t1_val

        def fcp(self, t, xx, yy):
            return self._grid.make_full0()

        def fT(self, t, xx, yy):
            return self._grid.make_full0()

        def fcl(self, t, xx, yy):
            return self._grid.make_full0()

        def fcd(self, t, xx, yy):
            return self._grid.make_full0()

        def fcs(self, t, xx, yy) -> np.ndarray:  # This is fcs_source
            # For corrector, we need fcs_source(t1)
            if not np.isclose(t, 0.0):  # t1 = t0 + dt
                return np.full(
                    self._grid.full_shape, fcs_source_t1_mock_val, dtype=float
                )
            return (
                self._grid.make_full0()
            )  # fcs_source(t0) is implicitly in Fcs_t0_mock_val

    # Setup specific mock field and forcing for this test
    mock_forcing = TempMockForcing(grid_fixture_ti, fcs_source_t1_mock_val)
    mock_field_for_corrector = TempMockFieldForCorrector(
        grid=grid_fixture_ti, model=integrator_hcs_ti._model, forcing_terms=mock_forcing
    )
    integrator_hcs_ti.semi_discrete_field = mock_field_for_corrector  # type: ignore

    at_t0_state: p1.StateVars = state_vars_ti_factory(cs_val=cs0_val)
    t0: float = 0.0

    # Setup Cl1, Cd1 to make the geometric part of R1 = Kd*(Sd-Cd1)*(1+Cl1) equal to 1
    cl1_val_test: float = 0.0
    # Use Sd from the integrator's model, which is model_fixture_ti
    cd1_val_test: float = integrator_hcs_ti._model.Sd - 1.0

    cl1_arr: np.ndarray = np.full(grid_fixture_ti.full_shape, cl1_val_test, dtype=float)
    cd1_arr: np.ndarray = np.full(grid_fixture_ti.full_shape, cd1_val_test, dtype=float)
    T1_dummy_arr: np.ndarray = grid_fixture_ti.make_full0()

    # Temporarily modify Kd on the integrator's model instance to achieve R1_target
    # R1 = 1 * Kd_modified = R1_target => Kd_modified = R1_target
    original_Kd_on_integrator_model = integrator_hcs_ti._model.Kd
    integrator_hcs_ti._model.Kd = (
        R1_target  # Assuming R1_target was chosen to make sense as a Kd
    )
    # (i.e., it's the K_d(Sd-Cd1)(1+Cl_1) value itself)

    cs1_corrected_masked: np.ndarray = integrator_hcs_ti.corrector_cs_step(
        T1_dummy_arr, cl1_arr, cd1_arr, at_t0=at_t0_state, t0=t0, dt=dt
    )

    integrator_hcs_ti._model.Kd = original_Kd_on_integrator_model  # Restore

    assert_allclose(
        cs1_corrected_masked[1:-1, 1:-1],
        expected_cs1_val_no_mask,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"corrector_cs_step failed: Y0_setup=({cs0_val},{Fcs_t0_mock_val},{fcs_source_t1_mock_val}), R1_target={R1_target}",
    )
    # Check boundary mask
    assert np.all(cs1_corrected_masked[0, :] == 0.0)
    assert np.all(cs1_corrected_masked[-1, :] == 0.0)
    assert np.all(cs1_corrected_masked[:, 0] == 0.0)
    assert np.all(cs1_corrected_masked[:, -1] == 0.0)


@pytest.mark.parametrize(
    "dt_val, R1_val_to_set_for_error",  # R1_val will be Kd * 1
    [
        (0.1, 20.0),  # dt*R1 = 0.1 * 20 = 2.0 (del_Y1 = 0)
        (0.1, 21.0),  # dt*R1 = 0.1 * 21 = 2.1 (del_Y1 < 0)
        (0.01, 200.0),  # dt*R1 = 0.01 * 200 = 2.0 (del_Y1 = 0)
        (0.01, 200.1),  # dt*R1 = 0.01 * 200.1 = 2.001 (del_Y1 < 0)
    ],
)
def test_corrector_cs_step_invalid_del_Y1_raises_error(
    integrator_hcs_ti: p1.P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple,
    state_vars_ti_factory: Callable[..., p1.StateVars],
    model_fixture_ti: p1.DefaultModel01,  # For Sd and restoring Kd
    grid_fixture_ti: p1.Grid,
    dt_val: float,
    R1_val_to_set_for_error: float,
):
    """
    Tests that corrector_cs_step raises ValueError if del_Y1 = (2 - dt*R1) is not positive.
    This test sets up Y0_term to be positive, so the problematic denominator is used.
    """
    cs0_val: float = 0.1  # Ensures Y0 will be positive with positive Fcs/fcs_source
    # Make Y0_term positive: Y0 = 2*Cs0 + dt*(Fcs_t0 + fcs_source_t1)
    # Let Fcs_t0_mock_val = 0.1, fcs_source_t1_mock_val = 0.1
    # Y0_term = 2*0.1 + dt_val*(0.1+0.1) = 0.2 + 0.2*dt_val > 0
    Fcs_t0_mock_val_setup: float = 0.1
    fcs_source_t1_mock_val_setup: float = 0.1

    class TempMockFieldForError(p1.SemiDiscreteField_HCsTriple):
        def Fcs(self, at_t_state_vars: p1.StateVars, t: float) -> np.ndarray:
            if np.isclose(t, 0.0):
                return np.full(self.grid.full_shape, Fcs_t0_mock_val_setup, dtype=float)
            return (
                self.grid.make_full0()
            )  # Should not be called for Fcs_star in corrector Y0

    class TempMockForcingError(p1.ForcingTermsBase):
        def __init__(self, grid: p1.Grid, fcs_t1_val: float):
            self._grid = grid
            self._fcs_t1_val = fcs_t1_val

        def fcp(self, t, xx, yy):
            return (
                self._grid.make_full0()
            )  # Other methods not strictly needed for this test

        def fT(self, t, xx, yy):
            return self._grid.make_full0()

        def fcl(self, t, xx, yy):
            return self._grid.make_full0()

        def fcd(self, t, xx, yy):
            return self._grid.make_full0()

        def fcs(self, t: float, xx, yy) -> np.ndarray:  # fcs_source
            if not np.isclose(t, 0.0):
                return np.full(
                    self._grid.full_shape, fcs_source_t1_mock_val_setup, dtype=float
                )
            return self._grid.make_full0()

    mock_forcing_err = TempMockForcingError(
        grid_fixture_ti, fcs_source_t1_mock_val_setup
    )
    mock_field_err = TempMockFieldForError(
        grid=grid_fixture_ti,
        model=integrator_hcs_ti._model,
        forcing_terms=mock_forcing_err,
    )
    integrator_hcs_ti.semi_discrete_field = mock_field_err  # type: ignore

    at_t0_state: p1.StateVars = state_vars_ti_factory(cs_val=cs0_val)
    t0: float = 0.0

    # Set Cl1, Cd1 to make geometric factor of R1 = 1
    cl1_val_test: float = 0.0
    cd1_val_test: float = integrator_hcs_ti._model.Sd - 1.0
    cl1_arr: np.ndarray = np.full(grid_fixture_ti.full_shape, cl1_val_test, dtype=float)
    cd1_arr: np.ndarray = np.full(grid_fixture_ti.full_shape, cd1_val_test, dtype=float)
    T1_dummy_arr: np.ndarray = grid_fixture_ti.make_full0()

    # Modify Kd to make dt*R1 >= 2
    original_Kd = integrator_hcs_ti._model.Kd
    integrator_hcs_ti._model.Kd = (
        R1_val_to_set_for_error  # This value is the target R1 as geom_factor=1
    )

    with pytest.raises(
        ValueError, match="Denominator 2 - Δt Kd .* below positiveness treshold"
    ):
        integrator_hcs_ti.corrector_cs_step(
            T1_dummy_arr, cl1_arr, cd1_arr, at_t0=at_t0_state, t0=t0, dt=dt_val
        )

    integrator_hcs_ti._model.Kd = original_Kd  # Restore

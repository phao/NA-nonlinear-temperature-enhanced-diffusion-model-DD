# test_semidiscrete_field_hcs_triple.py

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sympy  # Only needed if MMSCaseSymbolic is directly used here, not for these tests.

import prob1base as p1
from typing import Callable, Dict, Tuple, List, Any  # For type hints

# --- Test-Specific Model Constants ---
R0_test_sdf: float = 8.3144621
Ea_test_sdf: float = 1.60217662e-19
test_model_consts_sdf: p1.ModelConsts = p1.ModelConsts(
    R0=R0_test_sdf,
    Ea=Ea_test_sdf,
    K1=1.15e-2,
    K2=1.25e-2,
    K3=1.35e-2,
    K4=1.45e-2,
    DT=1.15e-3,
    Dl_max=8.15e-4,
    phi_l=1.15e-5,
    gamma_T=1.15e-9,
    Kd=1.15e-8,
    Sd=12,
    Dd_max=2.55e-6,
    phi_d=1.15e-5,
    phi_T=Ea_test_sdf / R0_test_sdf,
    r_sp=5.15e-2,
    T_ref=299,
)


# --- Fixtures ---
@pytest.fixture
def grid_fixture() -> p1.Grid:
    """Function-scoped grid fixture."""
    return p1.make_uniform_grid(N=4, M=4)


@pytest.fixture
def model_fixture_sdf() -> p1.DefaultModel01:
    """Fixture for the DefaultModel01 using test-specific constants for SDF tests."""
    return p1.DefaultModel01(mc=test_model_consts_sdf)


@pytest.fixture
def forcing_terms_mock(grid_fixture: p1.Grid) -> p1.ForcingTermsBase:
    """A simple mock for ForcingTermsBase that returns zeros for all forcing terms."""

    # Could have used p1.NoForcingTerms.
    class MockForcing(p1.ForcingTermsBase):
        def __init__(self, grid: p1.Grid):  # Added init for clarity
            self._grid = grid

        def fcp(self, t, xx, yy) -> np.ndarray:
            return self._grid.make_full0()

        def fT(self, t, xx, yy) -> np.ndarray:
            return self._grid.make_full0()

        def fcl(self, t, xx, yy) -> np.ndarray:
            return self._grid.make_full0()

        def fcd(self, t, xx, yy) -> np.ndarray:
            return self._grid.make_full0()

        def fcs(self, t, xx, yy) -> np.ndarray:
            return self._grid.make_full0()

    return MockForcing(grid_fixture)


@pytest.fixture
def semi_discrete_field_hcs(
    grid_fixture: p1.Grid,
    model_fixture_sdf: p1.DefaultModel01,
    forcing_terms_mock: p1.ForcingTermsBase,
) -> p1.SemiDiscreteField_HCsTriple:
    """Fixture for SemiDiscreteField_HCsTriple."""
    return p1.SemiDiscreteField_HCsTriple(
        grid=grid_fixture, model=model_fixture_sdf, forcing_terms=forcing_terms_mock
    )


@pytest.fixture
def state_vars_factory(
    grid_fixture: p1.Grid, model_fixture_sdf: p1.DefaultModel01
) -> Callable[..., p1.StateVars]:
    """Factory to create StateVars instances with specified cs values."""

    def _factory(
        cs_value_scalar: float,
        cp_val: float = 0.1,
        T_val: float = 0.2,
        cl_val: float = 0.3,
        cd_val: float = 0.4,
    ) -> p1.StateVars:
        return p1.StateVars(
            cp=np.full(grid_fixture.full_shape, cp_val, dtype=float),
            T=np.full(grid_fixture.full_shape, T_val, dtype=float),
            cl=np.full(grid_fixture.full_shape, cl_val, dtype=float),
            cd=np.full(grid_fixture.full_shape, cd_val, dtype=float),
            cs=np.full(grid_fixture.full_shape, cs_value_scalar, dtype=float),
            model=model_fixture_sdf,
            hh=grid_fixture.hh,
            kk=grid_fixture.kk,
        )

    return _factory


# --- Tests for SemiDiscreteField_HCsTriple ---


@pytest.mark.parametrize(
    "cs_input_scalar, expected_heaviside_factor",
    [
        (2.0, 1.0),
        (1e-10, 1.0),  # cs slightly > 0, (cs > 0) is True
        (0.0, 0.0),  # cs = 0, (cs > 0) is False
        (-1e-10, 0.0),  # cs slightly < 0, (cs > 0) is False
        (-2.0, 0.0),
    ],
)
def test_cscd_reaction_cs_method(
    semi_discrete_field_hcs: p1.SemiDiscreteField_HCsTriple,
    grid_fixture: p1.Grid,
    cs_input_scalar: float,
    expected_heaviside_factor: float,
) -> None:
    """
    Tests cscd_reaction_cs: model.Kd * (cs_arr > 0).
    """
    field_hcs: p1.SemiDiscreteField_HCsTriple = semi_discrete_field_hcs
    model: p1.DefaultModel01 = field_hcs.model

    cs_array_input: np.ndarray = np.full(
        grid_fixture.full_shape, cs_input_scalar, dtype=float
    )

    calculated_reaction_cs_arr: np.ndarray = field_hcs.cscd_reaction_cs(cs_array_input)

    expected_value_scalar: float = model.Kd * expected_heaviside_factor
    expected_array: np.ndarray = np.full(
        grid_fixture.full_shape, expected_value_scalar, dtype=float
    )

    assert_allclose(
        calculated_reaction_cs_arr,
        expected_array,
        rtol=1e-12,
        atol=1e-14,
        err_msg=f"cscd_reaction_cs failed for cs_input={cs_input_scalar}",
    )


@pytest.mark.parametrize(
    "cs_value_scalar, expected_heaviside_factor",
    [
        (1.0, 1.0),
        (1e-12, 1.0),  # Test value strictly positive, close to zero
        (0.0, 0.0),
        (-1e-12, 0.0),  # Test value strictly negative, close to zero
        (-1.0, 0.0),
    ],
)
def test_cscd_reaction_term_heaviside_effect(
    semi_discrete_field_hcs: p1.SemiDiscreteField_HCsTriple,
    state_vars_factory: Callable[..., p1.StateVars],
    model_fixture_sdf: p1.DefaultModel01,
    cs_value_scalar: float,
    expected_heaviside_factor: float,
) -> None:
    """
    Tests cscd_reaction_term, ensuring H(cs)=(cs > 0) propagates correctly.
    Formula: F1(cp) * (aT*T+bT) * (acl*cl+bcl) * (acd*cd+bcd) * (Kd * H(cs))
    """
    field_hcs: p1.SemiDiscreteField_HCsTriple = semi_discrete_field_hcs
    model: p1.DefaultModel01 = model_fixture_sdf

    cp_val_test, T_val_test, cl_val_test, cd_val_test = 0.1, 0.2, 0.3, 0.4
    current_state: p1.StateVars = state_vars_factory(
        cs_value_scalar,
        cp_val=cp_val_test,
        T_val=T_val_test,
        cl_val=cl_val_test,
        cd_val=cd_val_test,
    )

    term_calculated: np.ndarray = field_hcs.cscd_reaction_term(current_state)

    # Manually calculate the expected term factors
    # cscd_reaction_cp returns grid.const_with_nullbd(1)
    # For interior points, cp_factor_val will be 1.0
    a_T_val, b_T_val = field_hcs.cscd_reaction_T()  # Should be (0,1)
    a_cl_val, b_cl_val = field_hcs.cscd_reaction_cl()  # Should be (1,1)
    a_cd_val, b_cd_val = field_hcs.cscd_reaction_cd()  # Should be (-1, model.Sd)

    cp_factor_val_interior: float = 1.0
    T_factor_val_interior: float = (a_T_val * T_val_test) + b_T_val
    cl_factor_val_interior: float = (a_cl_val * cl_val_test) + b_cl_val
    cd_factor_val_interior: float = (a_cd_val * cd_val_test) + b_cd_val
    cs_heaviside_part_val_interior: float = model.Kd * expected_heaviside_factor

    expected_scalar_val_interior: float = (
        cp_factor_val_interior
        * T_factor_val_interior
        * cl_factor_val_interior
        * cd_factor_val_interior
        * cs_heaviside_part_val_interior
    )

    expected_term_array: np.ndarray = np.full(
        field_hcs.grid.full_shape, expected_scalar_val_interior, dtype=float
    )
    expected_term_array *= (
        field_hcs.grid.null_bd_mask
    )  # Mask applied by cscd_reaction_term

    assert_allclose(
        term_calculated,
        expected_term_array,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"cscd_reaction_term failed for cs={cs_value_scalar}",
    )


@pytest.mark.parametrize(
    "cs_value_scalar, expected_heaviside_factor",
    [
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, 0.0),
    ],
)
def test_Fcs_heaviside_effect(
    semi_discrete_field_hcs: p1.SemiDiscreteField_HCsTriple,
    state_vars_factory: Callable[..., p1.StateVars],
    cs_value_scalar: float,
    expected_heaviside_factor: float,
) -> None:
    """
    Tests Fcs = (fcs_source - cscd_reaction_term) * null_bd_mask.
    Mock fcs_source = 0. So, Fcs = -cscd_reaction_term * null_bd_mask.
    Since cscd_reaction_term is already masked, Fcs = -cscd_reaction_term.
    """
    field_hcs: p1.SemiDiscreteField_HCsTriple = semi_discrete_field_hcs
    current_state: p1.StateVars = state_vars_factory(cs_value_scalar)
    t_eval: float = 0.0

    grid = field_hcs.grid
    xx, yy = grid.xx, grid.yy
    
    fcs_source_mocked: np.ndarray = field_hcs.fcs(
        t_eval, xx, yy
    )  # Calls mock forcing_terms.fcs
    assert np.all(fcs_source_mocked == 0), "Mock forcing fcs should be zero."

    Fcs_calculated: np.ndarray = field_hcs.Fcs(current_state, t_eval)

    # Re-calculate expected cscd_reaction_term (already masked)
    model: p1.DefaultModel01 = field_hcs.model
    cp_val_test, T_val_test, cl_val_test, cd_val_test = 0.1, 0.2, 0.3, 0.4
    a_T_val, b_T_val = field_hcs.cscd_reaction_T()
    a_cl_val, b_cl_val = field_hcs.cscd_reaction_cl()
    a_cd_val, b_cd_val = field_hcs.cscd_reaction_cd()

    cp_factor_val_interior: float = 1.0
    T_factor_val_interior: float = (a_T_val * T_val_test) + b_T_val
    cl_factor_val_interior: float = (a_cl_val * cl_val_test) + b_cl_val
    cd_factor_val_interior: float = (a_cd_val * cd_val_test) + b_cd_val
    cs_heaviside_part_val_interior: float = model.Kd * expected_heaviside_factor

    expected_cscd_reaction_scalar_interior: float = (
        cp_factor_val_interior
        * T_factor_val_interior
        * cl_factor_val_interior
        * cd_factor_val_interior
        * cs_heaviside_part_val_interior
    )
    expected_cscd_reaction_array_interior: np.ndarray = np.full(
        field_hcs.grid.full_shape, expected_cscd_reaction_scalar_interior, dtype=float
    )
    expected_cscd_reaction_term_val: np.ndarray = (
        expected_cscd_reaction_array_interior * field_hcs.grid.null_bd_mask
    )

    # Expected Fcs = -expected_cscd_reaction_term_val (as source is 0 and outer mask is applied to already masked term)
    expected_Fcs_array: np.ndarray = -expected_cscd_reaction_term_val

    assert_allclose(
        Fcs_calculated,
        expected_Fcs_array,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"Fcs field calculation failed for cs={cs_value_scalar}",
    )


@pytest.mark.parametrize(
    "cs_value_scalar, expected_heaviside_factor",
    [
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, 0.0),
    ],
)
def test_Fcd_heaviside_effect(
    semi_discrete_field_hcs: p1.SemiDiscreteField_HCsTriple,
    state_vars_factory: Callable[..., p1.StateVars],
    cs_value_scalar: float,
    expected_heaviside_factor: float,
) -> None:
    """
    Tests Fcd_interior = Diffusive_Advective_Cd_terms_interior + cscd_reaction_term_interior + fcd_source_interior.
    Mock fcd_source = 0.
    """
    field_hcs: p1.SemiDiscreteField_HCsTriple = semi_discrete_field_hcs
    current_state: p1.StateVars = state_vars_factory(cs_value_scalar)
    t_eval: float = 0.0

    xx, yy = field_hcs.grid.xx, field_hcs.grid.yy
    fcd_source_mocked: np.ndarray = field_hcs.fcd(t_eval, xx, yy)
    assert np.all(fcd_source_mocked == 0)

    Fcd_calculated: np.ndarray = field_hcs.Fcd(current_state, t_eval)

    # Calculate expected cscd_reaction_term_val (already masked)
    model: p1.DefaultModel01 = field_hcs.model
    cp_val_test, T_val_test, cl_val_test, cd_val_test = 0.1, 0.2, 0.3, 0.4
    a_T_val, b_T_val = field_hcs.cscd_reaction_T()
    a_cl_val, b_cl_val = field_hcs.cscd_reaction_cl()
    a_cd_val, b_cd_val = field_hcs.cscd_reaction_cd()
    cp_factor_val_interior: float = 1.0
    T_factor_val_interior: float = (a_T_val * T_val_test) + b_T_val
    cl_factor_val_interior: float = (a_cl_val * cl_val_test) + b_cl_val
    cd_factor_val_interior: float = (a_cd_val * cd_val_test) + b_cd_val
    cs_heaviside_part_val_interior: float = model.Kd * expected_heaviside_factor
    expected_cscd_reaction_scalar_interior: float = (
        cp_factor_val_interior
        * T_factor_val_interior
        * cl_factor_val_interior
        * cd_factor_val_interior
        * cs_heaviside_part_val_interior
    )
    # cscd_reaction_term includes the null_bd_mask
    expected_cscd_reaction_term_val: np.ndarray = field_hcs.cscd_reaction_term(
        current_state
    )

    g: p1.Grid = field_hcs.grid
    diff_adv_cd_terms_interior: np.ndarray = (
        g.Dx_star(current_state.Dd_MxcpT * current_state.Dmxcd)
        + g.Dy_star(current_state.Dd_MycpT * current_state.Dmycd)
    )[1:-1, 1:-1]

    expected_Fcd_array: np.ndarray = (
        field_hcs.grid.make_full0()
    )  # Starts with fcd_source_mocked (0)
    expected_Fcd_array[1:-1, 1:-1] = (
        diff_adv_cd_terms_interior + expected_cscd_reaction_term_val[1:-1, 1:-1]
    )

    assert_allclose(
        Fcd_calculated,
        expected_Fcd_array,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"Fcd field calculation failed for cs={cs_value_scalar}",
    )


@pytest.mark.parametrize(
    "cs_value_scalar, expected_heaviside_factor",
    [
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, 0.0),
    ],
)
def test_delT_ab_cscd_reaction_ij_heaviside(
    semi_discrete_field_hcs: p1.SemiDiscreteField_HCsTriple,
    state_vars_factory: Callable[..., p1.StateVars],
    cs_value_scalar: float,
    expected_heaviside_factor: float,  # This factor isn't directly used if a_T=0
) -> None:
    """
    Tests delT_ab_cscd_reaction_ij for (a=0,b=0).
    For HCsTriple, cscd_reaction_T() -> (a_T=0, b_T=1).
    So, derivative w.r.t T_ij should be 0.
    """
    field_hcs: p1.SemiDiscreteField_HCsTriple = semi_discrete_field_hcs
    current_state: p1.StateVars = state_vars_factory(cs_value_scalar)

    delT_calc: np.ndarray = field_hcs.delT_ab_cscd_reaction_ij(current_state, a=0, b=0)

    a_T_val, _ = field_hcs.cscd_reaction_T()
    assert a_T_val == 0.0, "For HCsTriple, a_T (from cscd_reaction_T) must be 0."

    expected_delT: np.ndarray = field_hcs.grid.make_full0()

    assert_allclose(
        delT_calc,
        expected_delT,
        atol=1e-14,
        err_msg=f"delT_ab_cscd_reaction_ij failed for cs={cs_value_scalar}",
    )


@pytest.mark.parametrize(
    "cs_value_scalar, expected_heaviside_factor",
    [
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, 0.0),
    ],
)
def test_delcl_ab_cscd_reaction_ij_heaviside(
    semi_discrete_field_hcs: p1.SemiDiscreteField_HCsTriple,
    state_vars_factory: Callable[..., p1.StateVars],
    cs_value_scalar: float,
    expected_heaviside_factor: float,
) -> None:
    """
    Tests delcl_ab_cscd_reaction_ij for (a=0,b=0).
    Formula: cp_term * (a_T*T+b_T) * a_cl * (a_cd*cd+b_cd) * (Kd*H(cs)) * mask
    """
    field_hcs: p1.SemiDiscreteField_HCsTriple = semi_discrete_field_hcs
    current_state: p1.StateVars = state_vars_factory(cs_value_scalar)
    model: p1.DefaultModel01 = field_hcs.model

    delcl_calc: np.ndarray = field_hcs.delcl_ab_cscd_reaction_ij(
        current_state, a=0, b=0
    )

    cp_val_test, T_val_test, _cl_val_test_ignore, cd_val_test = (
        0.1,
        0.2,
        0.3,
        0.4,
    )  # From factory

    # Factors for manual calculation
    cp_term_val_interior: float = 1.0
    a_T_val, b_T_val = field_hcs.cscd_reaction_T()
    T_term_val_interior: float = a_T_val * T_val_test + b_T_val

    a_cl_val, _ = field_hcs.cscd_reaction_cl()  # This is the derivative part

    a_cd_val, b_cd_val = field_hcs.cscd_reaction_cd()
    cd_term_val_interior: float = a_cd_val * cd_val_test + b_cd_val

    cs_term_val_interior: float = model.Kd * expected_heaviside_factor

    expected_scalar_interior: float = (
        cp_term_val_interior
        * T_term_val_interior
        * a_cl_val
        * cd_term_val_interior
        * cs_term_val_interior
    )
    expected_delcl: np.ndarray = np.full(
        field_hcs.grid.full_shape, expected_scalar_interior, dtype=float
    )
    expected_delcl *= field_hcs.grid.null_bd_mask

    assert_allclose(
        delcl_calc,
        expected_delcl,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"delcl_ab_cscd_reaction_ij failed for cs={cs_value_scalar}",
    )


@pytest.mark.parametrize(
    "cs_value_scalar, expected_heaviside_factor",
    [
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, 0.0),
    ],
)
def test_delcd_ab_cscd_reaction_ij_heaviside(
    semi_discrete_field_hcs: p1.SemiDiscreteField_HCsTriple,
    state_vars_factory: Callable[..., p1.StateVars],
    cs_value_scalar: float,
    expected_heaviside_factor: float,
) -> None:
    """
    Tests delcd_ab_cscd_reaction_ij for (a=0,b=0).
    Formula: cp_term * (a_T*T+b_T) * (a_cl*cl+b_cl) * a_cd * (Kd*H(cs)) * mask
    """
    field_hcs: p1.SemiDiscreteField_HCsTriple = semi_discrete_field_hcs
    current_state: p1.StateVars = state_vars_factory(cs_value_scalar)
    model: p1.DefaultModel01 = field_hcs.model

    delcd_calc: np.ndarray = field_hcs.delcd_ab_cscd_reaction_ij(
        current_state, a=0, b=0
    )

    cp_val_test, T_val_test, cl_val_test, _cd_val_test_ignore = (
        0.1,
        0.2,
        0.3,
        0.4,
    )  # From factory

    cp_term_val_interior: float = 1.0
    a_T_val, b_T_val = field_hcs.cscd_reaction_T()
    T_term_val_interior: float = a_T_val * T_val_test + b_T_val

    a_cl_val, b_cl_val = field_hcs.cscd_reaction_cl()
    cl_term_val_interior: float = a_cl_val * cl_val_test + b_cl_val

    a_cd_val, _ = field_hcs.cscd_reaction_cd()  # This is the derivative part

    cs_term_val_interior: float = model.Kd * expected_heaviside_factor

    expected_scalar_interior: float = (
        cp_term_val_interior
        * T_term_val_interior
        * cl_term_val_interior
        * a_cd_val
        * cs_term_val_interior
    )
    expected_delcd: np.ndarray = np.full(
        field_hcs.grid.full_shape, expected_scalar_interior, dtype=float
    )
    expected_delcd *= field_hcs.grid.null_bd_mask

    assert_allclose(
        delcd_calc,
        expected_delcd,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"delcd_ab_cscd_reaction_ij failed for cs={cs_value_scalar}",
    )

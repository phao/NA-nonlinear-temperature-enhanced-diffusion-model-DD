# test_mms_trial_utils.py
import pytest
import numpy as np

import mms_trial_utils as mtu

# --- Fixtures for Mock Time Series Data ---


@pytest.fixture
def mock_error_time_series_t0_zero():
    """
    Mock time series where errors at t=0 are zero.
    - dt = 0.1
    - Variables: 'T' (integral), 'cp' (non-integral)
    """
    dt = 0.1
    # Data for t=0.0 (initial) - Errors are zero
    ts_data_0 = mtu.ErrorTimeSeries(
        t=0.0,
        h_norm_sq_errors={"T": 0.0, "cp": 0.0},
        grad_h_norm_p_sq_errors={"T": 0.0, "cp": 0.0},
    )
    # Data for t=0.1
    ts_data_1 = mtu.ErrorTimeSeries(
        t=0.1,
        h_norm_sq_errors={"T": 0.5, "cp": 0.1},  # H0^2(T,0.1)=0.5, H0^2(cp,0.1)=0.1
        grad_h_norm_p_sq_errors={"T": 2.0, "cp": 0.0},  # H1^2(T,0.1)=2
    )
    # Data for t=0.2
    ts_data_2 = mtu.ErrorTimeSeries(
        t=0.2,
        h_norm_sq_errors={"T": 0.2, "cp": 0.05},  # H0^2(T,0.2)=0.2, H0^2(cp,0.2)=0.05
        grad_h_norm_p_sq_errors={"T": 1.0, "cp": 0.0},  # H1^2(T,0.2)=1
    )
    return [ts_data_0, ts_data_1, ts_data_2], dt


@pytest.fixture
def mock_error_time_series_basic():
    """
    Provides a simple, short mock time series for testing NumericalErrorSummary.
    - 2 time steps (t=0, t=0.1) after initial t=0 data. Total 3 data points.
    - Variables: 'T' (integral), 'cp' (non-integral)
    """
    dt = 0.1
    # Data for t=0.0 (initial)
    # H0^2(T,0)=1, H1^2(T,0)=4
    # H0^2(cp,0)=0.25
    ts_data_0 = mtu.ErrorTimeSeries(
        t=0.0,
        h_norm_sq_errors={"T": 1.0, "cp": 0.25},
        grad_h_norm_p_sq_errors={"T": 4.0, "cp": 0.0},  # cp is non-integral
    )
    # Data for t=0.1
    # H0^2(T,0.1)=0.5, H1^2(T,0.1)=2
    # H0^2(cp,0.1)=0.1
    ts_data_1 = mtu.ErrorTimeSeries(
        t=0.1,
        h_norm_sq_errors={"T": 0.5, "cp": 0.1},
        grad_h_norm_p_sq_errors={"T": 2.0, "cp": 0.0},
    )
    # Data for t=0.2
    # H0^2(T,0.2)=0.2, H1^2(T,0.2)=1
    # H0^2(cp,0.2)=0.05
    ts_data_2 = mtu.ErrorTimeSeries(
        t=0.2,
        h_norm_sq_errors={"T": 0.2, "cp": 0.05},
        grad_h_norm_p_sq_errors={"T": 1.0, "cp": 0.0},
    )
    return [ts_data_0, ts_data_1, ts_data_2], dt


@pytest.fixture
def mock_error_time_series_single_step():
    """A time series with only initial and one computed step. dt=0.5"""
    dt = 0.5
    ts_data_0 = mtu.ErrorTimeSeries(
        t=0.0,
        h_norm_sq_errors={"T": 1.0, "cl": 0.8, "cp": 0.25},
        grad_h_norm_p_sq_errors={"T": 4.0, "cl": 2.0, "cp": 0.0},
    )
    ts_data_1 = mtu.ErrorTimeSeries(
        t=0.5,
        h_norm_sq_errors={"T": 0.5, "cl": 0.4, "cp": 0.1},
        grad_h_norm_p_sq_errors={"T": 2.0, "cl": 1.0, "cp": 0.0},
    )
    return [ts_data_0, ts_data_1], dt


@pytest.fixture
def mock_error_time_series_all_zero_grad():
    """Time series where gradient errors are always zero. dt=0.1"""
    dt = 0.1
    ts_data_0 = mtu.ErrorTimeSeries(
        t=0.0,
        h_norm_sq_errors={"T": 1.0, "cp": 0.25},
        grad_h_norm_p_sq_errors={"T": 0.0, "cp": 0.0},
    )
    ts_data_1 = mtu.ErrorTimeSeries(
        t=0.1,
        h_norm_sq_errors={"T": 0.5, "cp": 0.1},
        grad_h_norm_p_sq_errors={"T": 0.0, "cp": 0.0},
    )
    ts_data_2 = mtu.ErrorTimeSeries(
        t=0.2,
        h_norm_sq_errors={"T": 0.2, "cp": 0.05},
        grad_h_norm_p_sq_errors={"T": 0.0, "cp": 0.0},
    )
    return [ts_data_0, ts_data_1, ts_data_2], dt


# --- Tests for NumericalErrorSummary ---


def test_numerical_error_summary_instantiation_basic(mock_error_time_series_basic):
    """Test basic instantiation and that overall error is calculated."""
    time_series, dt = mock_error_time_series_basic
    variable_names = ["T", "cp"]
    integral_vars = ["T"]

    summary = mtu.NumericalErrorSummary(
        dt_used=dt,
        time_series_data=time_series,
        variable_names=variable_names,
        integral_vars=integral_vars,
    )
    assert summary is not None
    assert summary.overall_combined_error > 0
    assert "T" in summary.per_variable_sup_errors
    assert "cp" in summary.per_variable_sup_errors


def test_numerical_error_summary_overall_combined_error(mock_error_time_series_basic):
    """
    Verify the overall_combined_error calculation against manual calculation.
    Uses mock_error_time_series_basic: dt=0.1
    t=0: H0^2(T)=1, H0^2(cp)=0.25. H1^2(T)=4. SumH0^2 = 1.25. IntH1^2(0->0) = 0. Comb^2=1.25
    t=0.1: H0^2(T)=0.5, H0^2(cp)=0.1. H1^2(T)=2. SumH0^2 = 0.6.
           IntH1^2(0->0.1) = 0.5*0.1*(4+2) = 0.3. Comb^2 = 0.6 + 0.3 = 0.9
    t=0.2: H0^2(T)=0.2, H0^2(cp)=0.05. H1^2(T)=1. SumH0^2 = 0.25.
           IntH1^2(0.1->0.2) = 0.5*0.1*(2+1) = 0.15. Total IntH1^2(0->0.2) = 0.3 + 0.15 = 0.45
           Comb^2 = 0.25 + 0.45 = 0.7
    Max Comb^2 = 1.25. Expected overall error = sqrt(1.25)
    """
    time_series, dt = mock_error_time_series_basic
    variable_names = ["T", "cp"]
    integral_vars = ["T"]

    summary = mtu.NumericalErrorSummary(
        dt_used=dt,
        time_series_data=time_series,
        variable_names=variable_names,
        integral_vars=integral_vars,
    )

    # Manual calculation based on fixture data:
    # k=0 (t=0.0): sum_h0_sq = 1.0 (T) + 0.25 (cp) = 1.25
    #              int_h1_sq = 0.0 (as integral_vars=['T'], grad_h_norm_p_sq_errors for T is 4.0 at t=0)
    #              combined_sq_0 = 1.25 + 0 = 1.25
    # k=1 (t=0.1): sum_h0_sq = 0.5 (T) + 0.1 (cp) = 0.6
    #              running_int_h1_sq = 0.5 * 0.1 * (grad_h_norm_p_sq_errors['T'][t=0] + grad_h_norm_p_sq_errors['T'][t=0.1])
    #                                = 0.5 * 0.1 * (4.0 + 2.0) = 0.05 * 6.0 = 0.3
    #              combined_sq_1 = 0.6 + 0.3 = 0.9
    # k=2 (t=0.2): sum_h0_sq = 0.2 (T) + 0.05 (cp) = 0.25
    #              running_int_h1_sq_prev = 0.3
    #              running_int_h1_sq = running_int_h1_sq_prev + 0.5 * 0.1 * (grad_h_norm_p_sq_errors['T'][t=0.1] + grad_h_norm_p_sq_errors['T'][t=0.2])
    #                                = 0.3 + 0.5 * 0.1 * (2.0 + 1.0) = 0.3 + 0.05 * 3.0 = 0.3 + 0.15 = 0.45
    #              combined_sq_2 = 0.25 + 0.45 = 0.70
    # max_combined_norm_sq = max(1.25, 0.9, 0.70) = 1.25
    expected_overall_error = np.sqrt(1.25)

    assert summary.overall_combined_error == pytest.approx(expected_overall_error)


def test_numerical_error_summary_per_variable_integral_var(
    mock_error_time_series_basic,
):
    """
    Verify per-variable sup-norm for an integral variable ('T').
    For T (integral var): L_inf_t sqrt( H0_T^2(t) + integral_0^t H1_T^2(tau) dtau )
    t=0: H0_T^2=1, H1_T^2=4. IntH1_T^2=0. Term^2 = 1+0=1.
    t=0.1: H0_T^2=0.5, H1_T^2=2. IntH1_T^2(0->0.1) = 0.5*0.1*(4+2) = 0.3. Term^2 = 0.5+0.3=0.8
    t=0.2: H0_T^2=0.2, H1_T^2=1. IntH1_T^2(0.1->0.2) = 0.5*0.1*(2+1) = 0.15.
           Total IntH1_T^2(0->0.2) = 0.3+0.15 = 0.45. Term^2 = 0.2+0.45=0.65
    Max Term^2 = 1.0. Expected error for T = sqrt(1.0) = 1.0
    """
    time_series, dt = mock_error_time_series_basic
    variable_names = ["T", "cp"]
    integral_vars = ["T"]

    summary = mtu.NumericalErrorSummary(
        dt_used=dt,
        time_series_data=time_series,
        variable_names=variable_names,
        integral_vars=integral_vars,
    )

    # Manual calculation for 'T':
    # k=0 (t=0.0): h0_sq_T = 1.0. int_h1_sq_T = 0.0. combined_sq_T_0 = 1.0 + 0.0 = 1.0
    # k=1 (t=0.1): h0_sq_T = 0.5.
    #              running_int_h1_sq_T = 0.5 * 0.1 * (grad_H_T_sq[t=0] + grad_H_T_sq[t=0.1])
    #                                  = 0.5 * 0.1 * (4.0 + 2.0) = 0.3
    #              combined_sq_T_1 = 0.5 + 0.3 = 0.8
    # k=2 (t=0.2): h0_sq_T = 0.2.
    #              running_int_h1_sq_T_prev = 0.3
    #              running_int_h1_sq_T = running_int_h1_sq_T_prev + 0.5 * 0.1 * (grad_H_T_sq[t=0.1] + grad_H_T_sq[t=0.2])
    #                                  = 0.3 + 0.5 * 0.1 * (2.0 + 1.0) = 0.3 + 0.15 = 0.45
    #              combined_sq_T_2 = 0.2 + 0.45 = 0.65
    # max_norm_sq_T = max(1.0, 0.8, 0.65) = 1.0
    expected_error_T = np.sqrt(1.0)

    assert summary.per_variable_sup_errors["T"] == pytest.approx(expected_error_T)


def test_numerical_error_summary_per_variable_non_integral_var(
    mock_error_time_series_basic,
):
    """
    Verify per-variable sup-norm for a non-integral variable ('cp').
    For cp (non-integral var): L_inf_t L2_cp(t) = L_inf_t sqrt(H0_cp^2(t))
    H0_cp^2 values: 0.25 (t=0), 0.1 (t=0.1), 0.05 (t=0.2)
    Max H0_cp^2 = 0.25. Expected error for cp = sqrt(0.25) = 0.5
    """
    time_series, dt = mock_error_time_series_basic
    variable_names = ["T", "cp"]
    integral_vars = ["T"]

    summary = mtu.NumericalErrorSummary(
        dt_used=dt,
        time_series_data=time_series,
        variable_names=variable_names,
        integral_vars=integral_vars,
    )

    # Manual calculation for 'cp':
    # k=0 (t=0.0): h0_sq_cp = 0.25. combined_sq_cp_0 = 0.25 (integral_H1 part is 0)
    # k=1 (t=0.1): h0_sq_cp = 0.1.  combined_sq_cp_1 = 0.1
    # k=2 (t=0.2): h0_sq_cp = 0.05. combined_sq_cp_2 = 0.05
    # max_norm_sq_cp = max(0.25, 0.1, 0.05) = 0.25
    expected_error_cp = np.sqrt(0.25)

    assert summary.per_variable_sup_errors["cp"] == pytest.approx(expected_error_cp)


def test_numerical_error_summary_single_step_calc(mock_error_time_series_single_step):
    """Test with a time series having only one actual step after t=0."""
    time_series, dt = mock_error_time_series_single_step  # dt = 0.5
    variable_names = ["T", "cl", "cp"]
    integral_vars = ["T", "cl"]

    summary = mtu.NumericalErrorSummary(
        dt_used=dt,
        time_series_data=time_series,
        variable_names=variable_names,
        integral_vars=integral_vars,
    )

    # Overall:
    # t=0: SumH0^2 = H0_T^2(0)+H0_cl^2(0)+H0_cp^2(0) = 1.0+0.8+0.25 = 2.05. IntH1^2=0. Comb^2=2.05
    # t=0.5: SumH0^2 = H0_T^2(0.5)+H0_cl^2(0.5)+H0_cp^2(0.5) = 0.5+0.4+0.1 = 1.0
    #        IntH1^2_T = 0.5*0.5*(4+2)=1.5. IntH1^2_cl = 0.5*0.5*(2+1)=0.75. Total IntH1^2=1.5+0.75=2.25
    #        Comb^2 = 1.0 + 2.25 = 3.25
    # Max overall = sqrt(3.25)
    expected_overall = np.sqrt(3.25)
    assert summary.overall_combined_error == pytest.approx(expected_overall)

    # Per Var T:
    # t=0: H0_T^2=1. IntH1_T^2=0. Term^2=1
    # t=0.5: H0_T^2=0.5. IntH1_T^2(0->0.5) = 0.5*0.5*(4+2)=1.5. Term^2 = 0.5+1.5=2.0
    # Max T = sqrt(2.0)
    expected_T = np.sqrt(2.0)
    assert summary.per_variable_sup_errors["T"] == pytest.approx(expected_T)

    # Per Var cl:
    # t=0: H0_cl^2=0.8. IntH1_cl^2=0. Term^2=0.8
    # t=0.5: H0_cl^2=0.4. IntH1_cl^2(0->0.5) = 0.5*0.5*(2+1)=0.75. Term^2 = 0.4+0.75=1.15
    # Max cl = sqrt(1.15)
    expected_cl = np.sqrt(1.15)
    assert summary.per_variable_sup_errors["cl"] == pytest.approx(expected_cl)

    # Per Var cp:
    # t=0: H0_cp^2=0.25. Term^2=0.25
    # t=0.5: H0_cp^2=0.1. Term^2=0.1
    # Max cp = sqrt(0.25)
    expected_cp = np.sqrt(0.25)
    assert summary.per_variable_sup_errors["cp"] == pytest.approx(expected_cp)


def test_numerical_error_summary_all_zero_grad(mock_error_time_series_all_zero_grad):
    """Test case where all grad_h_norm_p_sq_errors are zero."""
    time_series, dt = mock_error_time_series_all_zero_grad  # dt = 0.1
    variable_names = ["T", "cp"]
    integral_vars = ["T"]  # T is integral, but its H1 error is zero

    summary = mtu.NumericalErrorSummary(
        dt_used=dt,
        time_series_data=time_series,
        variable_names=variable_names,
        integral_vars=integral_vars,
    )

    # Overall: Since H1 terms are zero, it becomes L_inf_t (Sum H0^2)
    # t=0: SumH0^2 = 1+0.25 = 1.25. IntH1^2=0. Comb^2=1.25
    # t=0.1: SumH0^2 = 0.5+0.1 = 0.6. IntH1^2=0. Comb^2=0.6
    # t=0.2: SumH0^2 = 0.2+0.05 = 0.25. IntH1^2=0. Comb^2=0.25
    # Max overall = sqrt(1.25)
    expected_overall = np.sqrt(1.25)
    assert summary.overall_combined_error == pytest.approx(expected_overall)

    # Per Var T: Since H1_T is zero, becomes L_inf_t L2_T
    # H0_T^2: 1.0, 0.5, 0.2. Max = 1.0
    expected_T = np.sqrt(1.0)
    assert summary.per_variable_sup_errors["T"] == pytest.approx(expected_T)

    # Per Var cp: (always L_inf_t L2_cp)
    # H0_cp^2: 0.25, 0.1, 0.05. Max = 0.25
    expected_cp = np.sqrt(0.25)
    assert summary.per_variable_sup_errors["cp"] == pytest.approx(expected_cp)


def test_numerical_error_summary_empty_time_series():
    """Test that an empty time series raises ValueError."""
    with pytest.raises(ValueError, match="time_series_data cannot be empty."):
        mtu.NumericalErrorSummary(
            dt_used=0.1, time_series_data=[], variable_names=["T"], integral_vars=["T"]
        )


def test_calculate_combined_error_norm_assertion_integral_subset_all():
    """Test assertion in calculate_combined_error_norm for integral_vars vs all_variables."""
    dt = 0.1
    ts_data = [
        mtu.ErrorTimeSeries(
            t=0.0, h_norm_sq_errors={"A": 1}, grad_h_norm_p_sq_errors={"A": 1}
        )
    ]

    # This is fine.
    mtu.calculate_combined_error_norm(
        ts_data, dt, integral_vars=[], all_variables=["A"]
    )

    with pytest.raises(
        AssertionError, match="integral_vars must be a subset of all_variables."
    ):
        mtu.calculate_combined_error_norm(
            ts_data, dt, integral_vars=["C"], all_variables=["A", "B"]
        )
    
    with pytest.raises(Exception):
        mtu.calculate_combined_error_norm(
            ts_data, dt, integral_vars=["A"], all_variables=["A", "B"]
        )


def test_numerical_error_summary_with_t0_zero_errors(mock_error_time_series_t0_zero):
    """
    Verify calculations when initial errors at t=0 are zero.
    dt=0.1
    Vars: T (integral), cp (non-integral)

    OverallCombinedError:
    t=0: H0_T^2=0, H0_cp^2=0. SumH0^2=0. IntH1_T^2=0. Comb^2=0.
    t=0.1: H0_T^2=0.5, H0_cp^2=0.1. SumH0^2=0.6.
           IntH1_T^2(0->0.1) = 0.5*0.1*(H1_T_sq(0) + H1_T_sq(0.1)) = 0.5*0.1*(0+2) = 0.1.
           Comb^2 = 0.6 + 0.1 = 0.7.
    t=0.2: H0_T^2=0.2, H0_cp^2=0.05. SumH0^2=0.25.
           IntH1_T^2(0.1->0.2) = 0.5*0.1*(H1_T_sq(0.1) + H1_T_sq(0.2)) = 0.5*0.1*(2+1) = 0.15.
           Total IntH1_T^2(0->0.2) = Int(0->0.1) + Int(0.1->0.2) = 0.1 + 0.15 = 0.25.
           Comb^2 = 0.25 + 0.25 = 0.5.
    Max Comb^2 = max(0, 0.7, 0.5) = 0.7. Expected overall = sqrt(0.7)

    PerVar 'T':
    t=0: H0_T^2=0. IntH1_T^2=0. Term^2=0.
    t=0.1: H0_T^2=0.5. IntH1_T^2(0->0.1) = 0.1. Term^2 = 0.5+0.1=0.6.
    t=0.2: H0_T^2=0.2. IntH1_T^2(0->0.2) = 0.25. Term^2 = 0.2+0.25=0.45.
    Max Term^2 = 0.6. Expected T_err = sqrt(0.6)

    PerVar 'cp':
    t=0: H0_cp^2=0. Term^2=0.
    t=0.1: H0_cp^2=0.1. Term^2=0.1.
    t=0.2: H0_cp^2=0.05. Term^2=0.05.
    Max Term^2 = 0.1. Expected cp_err = sqrt(0.1)
    """
    time_series, dt = mock_error_time_series_t0_zero
    variable_names = ["T", "cp"]
    integral_vars = ["T"]

    summary = mtu.NumericalErrorSummary(
        dt_used=dt,
        time_series_data=time_series,
        variable_names=variable_names,
        integral_vars=integral_vars,
    )

    expected_overall_error = np.sqrt(0.7)
    assert summary.overall_combined_error == pytest.approx(expected_overall_error)

    expected_error_T = np.sqrt(0.6)
    assert summary.per_variable_sup_errors["T"] == pytest.approx(expected_error_T)

    expected_error_cp = np.sqrt(0.1)
    assert summary.per_variable_sup_errors["cp"] == pytest.approx(expected_error_cp)

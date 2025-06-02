# mms_trial_utils.py
import math
import time
from typing import List, Dict, Tuple, Type, Optional, NamedTuple
import numpy as np
import prob1base as p1


class ErrorTimeSeries(NamedTuple):
    t: float
    h_norm_sq_errors: Dict[str, float]
    grad_h_norm_p_sq_errors: Dict[str, float]


def calculate_combined_error_norm(
    time_series_data: list[ErrorTimeSeries], dt: float, integral_vars: List[str], all_variables: Optional[List[str]] = None
) -> float:
    """Calculates the combined max-integral error norm from time series data."""
    max_combined_norm_sq = 0.0
    running_spatial_Hp_term = 0.0

    if all_variables is not None:
        assert all(ivar in all_variables for ivar in integral_vars), (
            "integral_vars must be a subset of all_variables."
        )

    integrand_values = []
    for step_data in time_series_data:
        step_integrand = sum(
            step_data.grad_h_norm_p_sq_errors[var] for var in integral_vars
        )
        integrand_values.append(step_integrand)

    for k, step_data in enumerate(time_series_data):
        if all_variables is None:
            spatial_H_term_sq = sum(
                step_data.h_norm_sq_errors.values()
            )  # Sum of the H-norms squared.
        else:
            spatial_H_term_sq = sum(
                step_data.h_norm_sq_errors[var] for var in all_variables
            )
        
        if k > 0:
            running_spatial_Hp_term += (
                0.5 * dt * (integrand_values[k - 1] + integrand_values[k])
            )  # Trapezoidal integration for the H1 part

        combined_norm_sq_tk = spatial_H_term_sq + running_spatial_Hp_term  # Combine
        max_combined_norm_sq = max(max_combined_norm_sq, combined_norm_sq_tk)

    final_error_norm = np.sqrt(max_combined_norm_sq)
    return final_error_norm


def run_simulation_collect_data(
    *,
    grid: p1.Grid,
    integrator: p1.TimeIntegratorBase,
    exact_sol_pack: p1.MMSCaseBase,
    initial_state: p1.StateVars,
    Tf: float,
    dt: float,
    t0: float = 0.0,
    variable_names: List[str],
    integral_vars: List[str],
) -> Tuple[List[ErrorTimeSeries], float]:
    """
    Runs the simulation from t=0 to Tf, collecting necessary data at each step
    for the combined error norm calculation.
    """
    current_t = t0
    current_num_state = initial_state
    time_series_data = []

    num_steps = math.ceil((Tf-t0) / dt)
    dt = (Tf - t0) / num_steps
    
    xx, yy = grid.xx, grid.yy

    def collect_errors(
        num_state: p1.StateVars, t: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Collects the H-norm and gradient H-norm errors at `t`.
        """
        exact_state_t = p1.state_from_mms_when(mms_case=exact_sol_pack, t=t, grid=grid)

        # Initialize error dictionaries
        h_norm_sq_errors = {}
        grad_h_norm_p_sq_errors = {}

        # Calculate errors for each variable
        for var_name in variable_names:
            num_sol = getattr(num_state, var_name)
            exact_sol = getattr(exact_state_t, var_name)
            error = num_sol - exact_sol
            h_norm_sq_errors[var_name] = grid.norm_H(error) ** 2

            if var_name in integral_vars:
                gradH_num = grid.grad_H(num_sol)
                gradH_ex = grid.grad_H(exact_sol)
                error_grad = (gradH_num[0] - gradH_ex[0], gradH_num[1] - gradH_ex[1])
                grad_h_norm_p_sq_errors[var_name] = (
                    grid.norm_p(error_grad[0], error_grad[1]) ** 2
                )
            else:
                grad_h_norm_p_sq_errors[var_name] = 0.0  # Convenience

        return h_norm_sq_errors, grad_h_norm_p_sq_errors
        # --- End of: collect_errors

    # --- Collect data at t=t0 ---

    h_norm_sq_errors_t0, grad_h_norm_p_sq_errors_t0 = collect_errors(
        initial_state, current_t
    )
    time_series_data.append(
        ErrorTimeSeries(
            t=current_t,
            h_norm_sq_errors=h_norm_sq_errors_t0,
            grad_h_norm_p_sq_errors=grad_h_norm_p_sq_errors_t0,
        )
    )

    for _step in range(num_steps):
        current_num_state = integrator.step(current_num_state, t0=current_t, dt=dt)
        current_t += dt

        # --- Collect data at current_t ---
        h_norm_sq_errors_tk, grad_h_norm_p_sq_errors_tk = collect_errors(
            current_num_state, current_t
        )
        time_series_data.append(
            ErrorTimeSeries(
                t=current_t,
                h_norm_sq_errors=h_norm_sq_errors_tk,
                grad_h_norm_p_sq_errors=grad_h_norm_p_sq_errors_tk,
            )
        )

    # Sanity check: ensure we reached the final time Tf
    assert np.isclose(
        current_t, Tf
    ), f"Final time mismatch: current_t={current_t}, Tf={Tf}"
   
    return time_series_data, dt


class NumericalErrorSummary:
    """
    Calculates and stores summary error norms from MMS time-series data.
    """

    def __init__(
        self,
        dt_used: float,
        time_series_data: List[ErrorTimeSeries],
        variable_names: List[str],
        integral_vars: List[str],
    ):
        """
        Args:
            dt_used (float): The actual time step used in the simulation.
            time_series_data (List[TimeStepData]): Raw data from run_simulation_collect_data.
            variable_names (List[str]): List of all variable names.
            integral_vars (List[str]): List of variable names for which H1 norm is relevant.
        """
        self.dt_used = dt_used
        self.variable_names = variable_names
        self.integral_vars = integral_vars

        if not time_series_data:
            raise ValueError("time_series_data cannot be empty.")

        # --- Calculate error norms ---

        # 1. Overall Combined Error Norm (L^inf_t H^1_x over all integral_vars)
        self.overall_combined_error: float = calculate_combined_error_norm(
            time_series_data, self.dt_used, self.integral_vars
        )

        # 2. Individual Sup-Norms for each variable
        self.per_variable_sup_errors: Dict[str, float] = {}

        for var_name in variable_names:
            intergal_variables = [] if var_name not in integral_vars else [var_name]
            self.per_variable_sup_errors[var_name] = calculate_combined_error_norm(
                time_series_data, self.dt_used, integral_vars=intergal_variables,
                all_variables=[var_name])

    def __repr__(self):
        per_var_repr = {k: f"{v:.4e}" for k, v in self.per_variable_sup_errors.items()}
        return (
            f"NumericalErrorSummary(dt={self.dt_used:.2e}, "
            f"OverallCombinedError={self.overall_combined_error:.4e}, "
            f"PerVariableSupErrors={per_var_repr})"
        )


class MMSTrial:
    """
    Encapsulates the setup and execution of a single Method of Manufactured Solutions trial.
    """

    def __init__(
        self,
        grid: p1.Grid,
        model: p1.DefaultModel01,
        mms_case_cls: Type[p1.MMSCaseBase],
        field_cls: Type[p1.SemiDiscreteFieldBase],
        forcing_terms_cls: Type[p1.ForcingTermsBase],
        integrator_cls: Type[p1.TimeIntegratorBase],
        mms_case_params: Optional[Dict] = {},
        integrator_params: Optional[Dict] = {},
        forcing_terms_params: Optional[Dict] = {},
        field_params: Optional[Dict] = {},
        variable_names: List[str] = None,
        integral_vars: List[str] = None,
    ):
        self.grid = grid
        # Ensure model is treated as immutable or a value copy is used internally by components
        # The check `other.model == self.model` for NamedTuple ModelConsts works for value equality.
        self.model = model
        self.mms_case_cls = mms_case_cls
        self.field_cls = field_cls
        self.forcing_terms_cls = forcing_terms_cls
        self.integrator_cls = integrator_cls

        self.variable_names = variable_names or ["cp", "T", "cl", "cd", "cs"]
        self.integral_vars = integral_vars or ["T", "cl", "cd"]

        # If *_cls types is more elaborate, dynamically create the class to fit needed format.

        self.mms_case = mms_case_cls(
            grid=self.grid, model=self.model, **mms_case_params
        )
        self.forcing_terms = forcing_terms_cls(
            mms_case=self.mms_case, model=self.model, **forcing_terms_params
        )
        self.field = field_cls(
            grid=self.grid,
            model=self.model,
            forcing_terms=self.forcing_terms,
            **field_params,
        )
        self.integrator = integrator_cls(
            semi_discrete_field=self.field, **integrator_params
        )
        self.initial_state = p1.state_from_mms_when(
            mms_case=self.mms_case, t=0.0, grid=self.grid
        )

    def run_for_errors(self, Tf: float, dt: float, t0: float = 0.0) -> NumericalErrorSummary:
        """
        Runs the simulation and computes error summaries.
        Args:
            Tf (float): Final time for the simulation.
            dt (float): Time step for the simulation.
            t0 (float): Initial time for the simulation, default is 0.0.
        """
        time_series_data, dt_adjusted = run_simulation_collect_data(
            grid=self.grid,
            integrator=self.integrator,
            exact_sol_pack=self.mms_case,
            initial_state=self.initial_state,
            Tf=Tf,
            dt=dt,
            t0=t0,
            variable_names=self.variable_names,
            integral_vars=self.integral_vars,
        )

        summary = NumericalErrorSummary(
            dt_used=dt_adjusted,
            time_series_data=time_series_data,
            variable_names=self.variable_names,
            integral_vars=self.integral_vars,
        )
        return summary

# prob1_mms_cases.py - v1

from typing import List
import numpy as np
import sympy

import prob1base as p1

from numpy import exp


class MMSCaseStiffExpDecay(p1.MMSCaseSymbolic):
    """
    MMS using a common spatial profile W(x,y) = x(1-x)y(1-y)
    but with different exponential decay rates exp(-a*t) for each variable
    to introduce stiffness and test time integration.

    Intended Stiffness Order (Fastest -> Slowest): Cl -> T -> Cd/Cs -> Cp (10x slower each)
    """

    def __init__(
        self,
        grid: p1.Grid,
        model: p1.DefaultModel01,
        *,
        a_base: float = 1.0,
    ):
        """
        Args:
            grid: Grid object.
            model: The Model object (used by the parent class).
            a_base: Base decay rate for the fastest component (cl).
                    A larger value increases stiffness and decay speed.
                    With a_base=1000, rates are:
                    a_cl=1000, a_T=100, a_cd=10, a_cs=10, a_cp=1.
        """
        # Define spatial part
        W_sym = p1.x_sym * (1 - p1.x_sym) * p1.y_sym * (1 - p1.y_sym)

        # Define decay rates based on 10x scaling
        a_cl = a_base
        a_T = a_base / 10.0
        a_cd = a_base / 100.0
        a_cs = a_base / 100.0  # Same timescale as Cd
        a_cp = a_base / 1000.0

        cp_sym = W_sym * sympy.exp(-a_cp * p1.t_sym)
        T_sym = W_sym * sympy.exp(-a_T * p1.t_sym)
        cl_sym = W_sym * sympy.exp(-a_cl * p1.t_sym)
        cd_sym = W_sym * sympy.exp(-a_cd * p1.t_sym)
        cs_sym = W_sym * sympy.exp(-a_cs * p1.t_sym)

        super().__init__(
            grid=grid,
            model=model,
            cp_sym_expr=cp_sym,
            T_sym_expr=T_sym,
            cl_sym_expr=cl_sym,
            cd_sym_expr=cd_sym,
            cs_sym_expr=cs_sym,
            t_var=p1.t_sym,  # Pass the symbols used
            x_var=p1.x_sym,
            y_var=p1.y_sym,
        )


def make_MMSCaseStiffExpDecay_cls(ref_speed):
    class the_MMSCaseStiffExpDecay(MMSCaseStiffExpDecay):
        def __init__(self, xx: np.ndarray, yy: np.ndarray, model: p1.DefaultModel01):
            super().__init__(xx=xx, yy=yy, model=model, a_base=ref_speed)

    return the_MMSCaseStiffExpDecay


# NOTE: Better name? Name should capture: polynomial in space, oscilating in time, positive and bounded profiles.
class MMSCasePolWithOscilatingTime(p1.MMSCaseSymbolic):
    """
    MMS using spatial profile

        W(x,y) = x (1 - x) y (1 - y)

    and an oscilating time profile:

        φ(t) = ampl * (1 + shrink * sin(speed * t)).

    All terms (cp, T, cl, cd, cs) are equal to this.

    Purpose: Oscilating time with simple space. Goal is high temporal error for large `speed` for stable spatial error (for fixed time). For positiveness, `shrink` should be in (0,1). By default, it goes as 1, which provides non-negativity.

    """

    def __init__(
        self,
        grid: p1.Grid,
        model: p1.DefaultModel01,
        *,
        ampl: float = 1,
        speed: float = 1,
        shrink: float = 1,
    ):
        """
        Initializes the MMS case with a polynomial spatial profile and an oscillating time profile.

        This constructor sets up the symbolic expressions for this profile and then calls the parent `MMSCaseSymbolic` initializer, which handles the symbolic differentiation and lambdification to provide the necessary functions for the MMS framework, and subsequently the forcing terms.

        Args:
            grid: Grid object to be used.
            model: An instance of `p1.DefaultModel01` (or a compatible model) containing the physical constants and model-specific functions (like Dl, V1, Dd) required by the parent class for calculating forcing terms.
            ampl: The amplitude of the temporal oscillation. The term `ampl` scales the entire profile. Defaults to 1.
            speed: The speed (angular frequency) of the `sin` term in the temporal oscillation `sin(speed * t)`. Defaults to 1.
            shrink: A factor that "shrinks" the sinusoidal part of the temporal oscillation relative to the constant part. `φ(t) = ampl * (1 + shrink * sin(speed * t))`. If `shrink` is in `(0, 1]`, and `ampl > 0`, then `φ(t)` (and thus the solution) will be non-negative. If `shrink > 1`, `φ(t)` can become negative. Defaults to 1.

        Note:
            There appears to be a potential mix-up in the `super().__init__` call regarding `t_var`, `x_var`, and `y_var`. They are currently passed as `t_var=x, x_var=y, y_var=t`. Given the symbolic variables `x, y, t` are imported from `prob1base` and used to construct `the_profile`, it should likely be `t_var=t, x_var=x, y_var=y` to match the symbols used in the expressions.
        """
        from prob1base import x_sym as x, y_sym as y, t_sym as t
        from sympy import sin

        # Define spatial and temporal parts.
        W = x * (1 - x) * y * (1 - y)
        phi = ampl * (1 + shrink * sin(speed * t))
        the_profile = phi * W

        super().__init__(
            grid=grid,
            model=model,
            cp_sym_expr=the_profile,
            T_sym_expr=the_profile,
            cl_sym_expr=the_profile,
            cd_sym_expr=the_profile,
            cs_sym_expr=the_profile,
            t_var=t,
            x_var=x,
            y_var=y,
        )


def make_MMSCasePolWithOscilatingTime_cls(*, ampl, speed):
    class the_MMSCasePolWithOscilatingTime(MMSCasePolWithOscilatingTime):
        def __init__(self, grid: p1.Grid, model: p1.DefaultModel01):
            super().__init__(
                grid=grid,
                model=model,
                ampl=ampl,
                speed=speed,
            )

    return the_MMSCasePolWithOscilatingTime


class MMSCaseSlowlyChangingPeaks(p1.MMSCaseSymbolic):
    """
    MMS using spatial profile

        W(x,y) = Const (x² + y²)³ sin(πx) sin(πy)

    and a very slow moving temporal profile

        φ(t) = exp(-a t)

    for some small abs(a). All terms (cp, T, cl, cd, cs) are equal to this.

    Purpose: small temporal error, large spatial error.
    """

    def __init__(
        self,
        grid: p1.Grid,
        model: p1.DefaultModel01,
        *,
        leading_spatial_const=1e1,
        evol_speed: float = 1e-1,
    ):
        """
        Args:
            grid (Grid): the grid..
            model: The Model object (used by the parent class).
            leading_spatial_const: `Const` in class docstring
            evol_speed: `a` in class docstring
        """
        # Define spatial part
        norm_sym = p1.x_sym**2 + p1.y_sym**2
        trig_sym = sympy.sin(sympy.pi * p1.x_sym) * sympy.sin(sympy.pi * p1.y_sym)
        W_sym = norm_sym**3 * trig_sym * leading_spatial_const
        phi = sympy.exp(-evol_speed * p1.t_sym)
        f = W_sym * phi

        super().__init__(
            grid=grid,
            model=model,
            cp_sym_expr=f,
            T_sym_expr=f,
            cl_sym_expr=f,
            cd_sym_expr=f,
            cs_sym_expr=f,
            t_var=p1.t_sym,  # Pass the symbols used
            x_var=p1.x_sym,
            y_var=p1.y_sym,
        )


def make_MMSCaseSlowlyChangingPeaks_cls(*, leading_spatial_const, evol_speed):
    class the_MMSCaseSlowlyChangingPeaks(MMSCaseSlowlyChangingPeaks):
        def __init__(self, grid: p1.Grid, model: p1.DefaultModel01):
            super().__init__(
                grid=grid,
                model=model,
                evol_speed=evol_speed,
                leading_spatial_const=leading_spatial_const,
            )

    return the_MMSCaseSlowlyChangingPeaks


MMSCaseSlowlyChangingPeaks_Slow1e1 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e-1
)
MMSCaseSlowlyChangingPeaks_Slow1e2 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e-2
)
MMSCaseSlowlyChangingPeaks_Slow1e3 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e-3
)
MMSCaseSlowlyChangingPeaks_Slow1e4 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e-4
)
MMSCaseSlowlyChangingPeaks_Slow1e8 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e-8
)
MMSCaseSlowlyChangingPeaks_Slow1e16 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e-16
)
MMSCaseSlowlyChangingPeaks_Fast1e1 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e1
)
MMSCaseSlowlyChangingPeaks_Fast1e2 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e2
)
MMSCaseSlowlyChangingPeaks_Fast1e3 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e3
)
MMSCaseSlowlyChangingPeaks_Fast1e4 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e4
)
MMSCaseSlowlyChangingPeaks_Fast1e8 = make_MMSCaseSlowlyChangingPeaks_cls(
    leading_spatial_const=1.0, evol_speed=1e8
)


class MMSCasePol(p1.MMSCaseSymbolic):
    """
    MMS with exact polynomial solution
    We'll start with all of them being:

        f(t, x, y) = x * (1 - x) * y * (1 - y) / (1 + t)
    """

    def __init__(self, grid: p1.Grid, *, model: p1.DefaultModel01):
        # Define spatial part
        W_sym = p1.x_sym * (1 - p1.x_sym) * p1.y_sym * (1 - p1.y_sym)
        t_denom_sym = 1 + p1.t_sym

        # Define symbolic expression, f, for the variables
        f_sym = W_sym / t_denom_sym

        super().__init__(
            grid=grid,
            model=model,
            cp_sym_expr=f_sym,
            T_sym_expr=f_sym,
            cl_sym_expr=f_sym,
            cd_sym_expr=f_sym,
            cs_sym_expr=f_sym,
            t_var=p1.t_sym,  # Pass the symbols used
            x_var=p1.x_sym,
            y_var=p1.y_sym,
        )


class MMSCaseExpSin(p1.MMSCaseSymbolic):
    """
    Objeto com funções e dados exatos. Inclui solução exata e também funções
    de modelo. Aqui, as soluções são:

        Cp(t) = Cp(0) exp(∫₀ᵗ (-K1 (1 + Cl(s)) - K2 T(s)) ds)
        T(t) = exp(-2 π² D_T) W
        Cl(t) = -exp(-t) * W
        Cd(t) = -Cl(t) = exp(-t) W
        Cs(t) = Cs(0) exp(∫₀ᵗ (- Kd (Sd - Cd(s)) (1 + Cl(s)) ds)

    sendo

        W(x,y) = Sin(π x) Sin(π y)
    """

    def __init__(self, grid: p1.Grid, *, model: p1.DefaultModel01):
        x = p1.x_sym
        y = p1.y_sym
        t = p1.t_sym

        from sympy import sin, exp, pi, integrate

        K1 = model.K1
        K2 = model.K2
        Sd = model.Sd
        Kd = model.Kd
        DT = model.DT

        pi2 = pi**2

        W = sin(pi * x) * sin(pi * y)
        T = exp(-2 * pi2 * DT * t) * W
        cl = -exp(-t) * W
        cd = -cl

        # TODO: as sympy.integrate can be slow, and these can be integrated using a symbolic K1, K2, Kd, Sd, we should do this integration once (at a class setup level) and just substitute here for an actual model's K1, K2, K3, Sd.

        cp_exp_prim = integrate(-K1 * (1 + cl) - K2 * T, t)
        cp0 = W
        cp = cp0 * exp(cp_exp_prim - cp_exp_prim.subs(t, 0))

        cs_exp_prim = integrate(-Kd * (Sd - cd) * (1 + cl), t)
        cs0 = model.r_sp * cp0
        cs = cs0 * exp(cs_exp_prim - cs_exp_prim.subs(t, 0))

        super().__init__(
            grid=grid,
            model=model,
            cp_sym_expr=cp,
            T_sym_expr=T,
            cl_sym_expr=cl,
            cd_sym_expr=cd,
            cs_sym_expr=cs,
            t_var=t,
            x_var=x,
            y_var=y,
        )


# --- MMS Case designed to have Cs cross zero ---
class MMSCaseCsZeroCrossing(p1.MMSCaseSymbolic):
    """
    MMS using constant zero for Cp, T, Cl, Cd.
    Cs(t,x,y) uses a function designed to cross zero during the simulation.
    Example uses a linear ramp: Cs(t,x,y) = (A - B*t) * W(x,y)
    """

    def __init__(
        self,
        grid: p1.Grid,
        model: p1.DefaultModel01,
        *,
        cs_A: float = 0.5,  # Initial value multiplier at t=0
        cs_B: float = 1.0,  # Determines slope/crossing time t_cross = A/B
        spatial_profile_expr: sympy.Expr = (
            p1.x_sym * (1 - p1.x_sym) * p1.y_sym * (1 - p1.y_sym)
        ),
    ):
        """
        Args:
            grid: the Grid.
            model: Model instance.
            cs_A: Amplitude factor for Cs at t=0.
            cs_B: Coefficient for linear time decay term for Cs.
            spatial_profile_expr: Spatial profile W(x,y).
        """
        # Ensure parameters are symbolic floats for SymPy expressions
        cs_A_sym = sympy.Float(cs_A)
        cs_B_sym = sympy.Float(cs_B)

        # Manufactured solutions: Cp=T=Cl=Cd=0
        cp_ex_sym = sympy.S(0)
        T_ex_sym = sympy.S(0)
        cl_ex_sym = sympy.S(0)
        cd_ex_sym = sympy.S(0)

        # Cs uses a linear ramp to ensure zero crossing if B > 0
        # cs(t,x,y) = (A - B*t) * W(x,y)
        cs_t_ramp_sym = cs_A_sym - cs_B_sym * p1.t_sym
        cs_ex_sym = cs_t_ramp_sym * spatial_profile_expr

        # --- Alternative Cs form: Exponential difference ---
        # k1 = 0.1
        # k2 = 0.5
        # cs_A_alt = 1.0
        # cs_B_alt = 1.1 # Need B > A for crossing
        # cs_ex_sym = ( sympy.Float(cs_A_alt) * sympy.exp(-k1 * p1.t_sym) -
        #               sympy.Float(cs_B_alt) * sympy.exp(-k2 * p1.t_sym)
        #             ) * spatial_profile_expr
        # ----------------------------------------------------

        super().__init__(
            grid=grid,
            model=model,
            cp_sym_expr=cp_ex_sym,
            T_sym_expr=T_ex_sym,
            cl_sym_expr=cl_ex_sym,
            cd_sym_expr=cd_ex_sym,
            cs_sym_expr=cs_ex_sym,
            t_var=p1.t_sym,
            x_var=p1.x_sym,
            y_var=p1.y_sym,
        )


class MMSCaseNonFullySmoothPol(p1.MMSCaseSymbolic):
    """
    MMS using a non-fully smooth polynomial solution.
    The solution is defined as:

        f(t, x, y) = φ(t) W(x, y) * |(x - θ) * (y - θ)| ** γ
        where
            φ(t) = 1 / (1 + t)
            W(x, y) = x * (1 - x) * y * (1 - y)
            θ is a number in (0, 1)
            γ is a parameter that controls the non-smoothness of the solution.

    It is known that for γ ∈ (2, 3], this solution is in H² but not in H³.
    Similarly, for γ ∈ (3, 4], it is in H³ but not in H⁴.

    Each component solution (cp, T, cl, cd, cs) will be an expression of this form, each with its own γ.

    θ defaults to 0.5.
    """

    def __init__(self, grid: p1.Grid, *, model: p1.DefaultModel01, gamma: List[float], theta: float = 1/np.pi):
        """
        Args:
            grid: The grid object.
            model: The model object (used by the parent class).
            gamma: The exponent-vector for the non-smoothness. Each entry should be in (2, 3] for H² but not H³, or (3, 4] for H³ but not H⁴.
        """
        from prob1base import x_sym, y_sym, t_sym

        # If symbols aren't properly constrained, this will raise an error.
        # Ensure that x_sym, y_sym, t_sym are all real and non-negative.
        # This is needed so that sympy can handle differentiating the symbolic expressions needed for the MMS (otherwise, sympy might think I'm in a more general setting than I actually am).
        if not (x_sym.is_real and y_sym.is_real and t_sym.is_real):
            raise ValueError("x_sym, y_sym, and t_sym must be real symbols.")
        if not (x_sym.is_nonnegative and y_sym.is_nonnegative and t_sym.is_nonnegative):
            raise ValueError("x_sym, y_sym, and t_sym must be non-negative symbols.")
        
        if np.isscalar(gamma):
            gamma = [float(gamma)]
        
        assert isinstance(gamma, list), "gamma must be a single number or a list of numbers."
        assert len(gamma) in [
            1,
            2,
            5,
        ], "gamma must be a list of length 1, 2, or 5 numbers: one for all (cp, T, cl, cd, cs); two for (cp, cs) and (T, cl, cd); or five for cp,T, cl, cd, cs."

        len_gamma = len(gamma)
        if len_gamma == 1:
            gamma = [gamma[0]] * 5
        elif len_gamma == 2:
            gamma = [gamma[0], gamma[1], gamma[1], gamma[1], gamma[0]]
        elif len_gamma == 5:
            pass
        else:
            raise ValueError("Invalid length for gamma. Must be 1, 2, or 5.")
        
        assert all(gamma[j] > 1 for j in [0, 4]), "Cp's and cs' gamma (0, 4) must be greater than 1 for minimal smoothness."
        assert all(gamma[j] > 2 for j in [1, 2, 3]), "T's, cl's, and cd's gammas (1, 2, 3) must be greater than 2 for minimal smoothness."

        assert 0 < theta < 1, "Theta must be in (0, 1)."

        # Define spatial part
        W_sym = x_sym * (1 - x_sym) * y_sym * (1 - y_sym)
        phi_sym = 1 / (1 + t_sym)

        # Non-smoothness factor
        non_smooth_factor_base = sympy.Abs((x_sym - theta) * (y_sym - theta))
        non_smooth_factor_cp = non_smooth_factor_base ** gamma[0]
        non_smooth_factor_T = non_smooth_factor_base ** gamma[1]
        non_smooth_factor_cl = non_smooth_factor_base ** gamma[2]
        non_smooth_factor_cd = non_smooth_factor_base ** gamma[3]
        non_smooth_factor_cs = non_smooth_factor_base ** gamma[4]

        # Full expression
        common_expr = phi_sym * W_sym
        cp_exact = common_expr * non_smooth_factor_cp
        T_exact = common_expr * non_smooth_factor_T
        cl_exact = common_expr * non_smooth_factor_cl
        cd_exact = common_expr * non_smooth_factor_cd
        cs_exact = common_expr * non_smooth_factor_cs

        super().__init__(
            grid=grid,
            model=model,
            cp_sym_expr=cp_exact,
            T_sym_expr=T_exact,
            cl_sym_expr=cl_exact,
            cd_sym_expr=cd_exact,
            cs_sym_expr=cs_exact,
            t_var=t_sym,  # Pass the symbols used
            x_var=x_sym,
            y_var=y_sym,
        )

def make_MMSCaseNonFullySmoothPol_cls(gamma):
    class the_MMSCaseNonFullySmoothPol(MMSCaseNonFullySmoothPol):
        def __init__(self, grid: p1.Grid, model: p1.DefaultModel01):
            super().__init__(grid=grid, model=model, gamma=gamma)

    return the_MMSCaseNonFullySmoothPol

MMSCaseNonFullySmoothPol_cpcsH2_TclcdH3 = make_MMSCaseNonFullySmoothPol_cls(gamma=[2.1, 3.1])
MMSCaseNonFullySmoothPol_cpcsH1_TclcdH2 = make_MMSCaseNonFullySmoothPol_cls(gamma=[1.1, 2.1])
MMSCaseNonFullySmoothPol_cpcsH2_TclcdH2 = make_MMSCaseNonFullySmoothPol_cls(gamma=2.1)
MMSCaseNonFullySmoothPol_cpcsH3_TclcdH4 = make_MMSCaseNonFullySmoothPol_cls(gamma=[3.1, 4.1])
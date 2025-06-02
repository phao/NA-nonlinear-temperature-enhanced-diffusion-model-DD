# prob1base.py - v1

# TODO: There is currently a certain strangeness dependency on grid. For example, MMSCaseBase shouldn't need grids. Check forcing terms and semidiscrete field types as well to see if they really need grids.

"""
Legenda: p1: +1; m1: -1
Exemplo: ip1j: i+1,j; ijm1: i,j-1
"""

from collections import namedtuple

import numbers  # To check for standard Python numeric types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sympy

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, Literal, List, Dict, NamedTuple

from numpy import exp
from sympy import DiracDelta

Diag5 = namedtuple("Diag5", ["sub", "rspec", "cspec"])

class ModelConsts(NamedTuple):
    R0: float
    Ea: float
    K1: float
    K2: float
    K3: float
    K4: float
    DT: float
    Dl_max: float
    phi_l: float
    gamma_T: float
    Kd: float
    Sd: float
    Dd_max: float
    phi_d: float
    phi_T: float
    r_sp: float
    T_ref: float = 300


R0 = 8.3144621
Ea = 1.60217662e-19
default_model_consts = ModelConsts(
    R0=R0,
    Ea=Ea,
    K1=1e-2,
    K2=1e-2,
    K3=1e-2,
    K4=1e-2,
    DT=1e-3,
    Dl_max=8.01e-4,
    phi_l=1e-5,
    gamma_T=1e-9,
    Kd=1e-8,
    Sd=10,
    Dd_max=2.46e-6,
    phi_d=1e-5,
    phi_T=Ea / R0,
    r_sp=5e-2,
    T_ref=300,
)


class DefaultModel01:
    def __init__(self, mc: ModelConsts):
        for k, v in mc._asdict().items():
            setattr(self, k, v)

    def with_changes(self, **kwargs):
        output_base = DefaultModel01(
            ModelConsts(**{k: getattr(self, k) for k in ModelConsts._fields})
        )
        for k, v in kwargs.items():
            setattr(output_base, k, v)
        return output_base

    def copy(self):
        return self.with_changes()

    # def Dl(self, cp, *, d=0):
    #     """
    #     Dl = Dl_max exp(-phi_l cp)
    #     """
    #     phi_l = self.phi_l
    #     Dl_max = self.Dl_max
    #     the_Dl = Dl_max * exp(-phi_l * cp)
    #     return ((-phi_l) ** d) * the_Dl

    def Dl(self, cp, *, d=0):
        """
        Dl = Dl_max exp(-phi_l cp)
        Works with both NumPy arrays and SymPy expressions for cp.
        """
        phi_l = self.phi_l
        Dl_max = self.Dl_max

        is_symbolic = isinstance(cp, sympy.Expr)
        exp_func = sympy.exp if is_symbolic else np.exp
        the_Dl_base = Dl_max * exp_func(-phi_l * cp)

        if is_symbolic:
            return sympy.diff(the_Dl_base, cp, d)
        else:
            # -- Compute numerical derivative, up to order 2.
            return ((-phi_l) ** d) * the_Dl_base

    def V1(self, T, *, d=0):
        """
        V1 = gammaT * T
        Works with both NumPy arrays and SymPy expressions.
        """

        is_symbolic = isinstance(T, sympy.Expr)

        if is_symbolic:
            return sympy.diff(self.gamma_T * T, T, d)
        else:
            gamma_T = self.gamma_T
            if d == 0:
                return gamma_T * T
            elif d == 1:
                return gamma_T * np.ones_like(T)
            else:
                return np.zeros_like(T)

    def V2(self, T, *, d=0):
        is_symbolic = isinstance(T, sympy.Expr)
        if is_symbolic:
            return sympy.S(0)
        else:
            return np.zeros_like(T)

    def Dd(self, cp, T, *, d=(0, 0)):
        """
        Dd = Dd_max * exp(-phi_d * cp) * exp(-phi_T / T)

        Note: The function f(x) = exp(-a/x), where a > 0 is a constant,
        should be evaluated as zero when x = 0 (taking the limit as x approaches 0).

        Works with both NumPy arrays and SymPy expressions.

        Args:
            cp: Concentration values, can be NumPy array or SymPy expression
            T: Temperature values, can be NumPy array or SymPy expression
            d: Tuple (d_cp, d_T) specifying derivative orders with respect to cp and T.
               Default is (0, 0) for no derivatives.

        Returns:
            NumPy array or SymPy expression: The diffusion coefficient Dd or its derivatives.
            For NumPy arrays, returns zeros where T=0 to handle the exp(-phi_T/T) singularity.
            For SymPy expressions, returns the symbolic derivative.

        Raises:
            AssertionError: If cp and T have different symbolic types (both should be symbolic or both numeric to pass this check).
        """

        is_symbolic_cp = isinstance(cp, sympy.Expr)
        is_symbolic_T = isinstance(T, sympy.Expr)

        assert (is_symbolic_cp and is_symbolic_T) or (
            (not is_symbolic_cp) and (not is_symbolic_T)
        )

        is_symbolic = is_symbolic_cp and is_symbolic_T

        if is_symbolic:
            sym_Dd = (
                self.Dd_max * sympy.exp(-self.phi_d * cp) * sympy.exp(-self.phi_T / T)
            )
            del_cp_Dd = sympy.diff(sym_Dd, cp, d[0])
            del_T_del_cp_Dd = sympy.diff(del_cp_Dd, T, d[1])
            return del_T_del_cp_Dd

        else:
            cp = np.asarray(cp)
            T = np.asarray(T)

            assert cp.shape == T.shape

            Dd_max = self.Dd_max
            phi_d = self.phi_d
            phi_T = self.phi_T

            Tnz = T != 0
            the_Dd = np.zeros_like(T, dtype=np.float64)
            the_Dd[Tnz] = Dd_max * exp(-phi_d * cp[Tnz]) * exp(-phi_T / T[Tnz])

            if d == (0, 0):
                return the_Dd
            elif d == (1, 0):
                return -phi_d * the_Dd
            elif d == (0, 1):
                the_dT_Dd = the_Dd
                the_dT_Dd[Tnz] *= phi_T / (T[Tnz] ** 2)
                return the_dT_Dd


class DefaultModel02(DefaultModel01):
    def __init__(self, mc: ModelConsts):
        super().__init__(mc)

    def Dd(self, cp, T, *, d=(0, 0)):
        """
        Dd
            = super().Dd(cp, T + T_Ref)
            = Dd_max * exp(-phi_d cp) * exp(-phi_T/(T + T_ref0))

        Works with both NumPy arrays and SymPy expressions.
        """
        return super().Dd(cp, T + self.T_ref, d=d)


class Grid:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # NOTE: can work with simple 1D x,y vectors, but also with 2D meshgrid'ed versions as well. Dimension check (len of shape) is used to determine which is which.

        x_dim = len(x.shape)
        y_dim = len(y.shape)
        assert y_dim == x_dim
        assert x_dim in [1, 2], "Grid: x,y's shape must be 1D or 2D."

        if x_dim == 2:
            assert (
                x.shape == y.shape
            ), "Grid: for meshgrid'ed x,y, same shape is required."
            x = x[:, 0]
            y = y[0, :]

        Np1 = len(x)
        Mp1 = len(y)
        N = Np1 - 1
        M = Mp1 - 1
        self.M, self.N, self.x, self.y = M, N, x, y

        # NOTE: A ordem 'ij' é por eu ter usado i para o índice em x e j para o índice em y.
        xx, yy = np.meshgrid(x, y, indexing="ij")

        self.xx, self.yy = xx, yy

        assert xx.shape == (N + 1, M + 1)
        assert yy.shape == (N + 1, M + 1)

        xx_phalf = np.zeros((N + 1, M + 1))
        xx_phalf[:-1, :] = 0.5 * (xx[:-1, :] + xx[1:, :])
        yy_phalf = np.zeros((N + 1, M + 1))
        yy_phalf[:, :-1] = 0.5 * (yy[:, :-1] + yy[:, 1:])

        assert all(
            xx_phalf[i, j] == xx_phalf[i, 0]
            and xx_phalf[i, j] == (xx[i, j] + xx[i + 1, j]) * 0.5
            and xx_phalf[i, j] == (x[i] + x[i + 1]) * 0.5
            for i in range(N - 1)
            for j in range(M)
        )

        assert all(
            yy_phalf[i, j] == yy_phalf[0, j]
            and yy_phalf[i, j] == (yy[i, j] + yy[i, j + 1]) * 0.5
            and yy_phalf[i, j] == (y[j] + y[j + 1]) * 0.5
            for i in range(N)
            for j in range(M - 1)
        )

        self.xx_phalf = xx_phalf
        self.yy_phalf = yy_phalf

        """
        h[i] deve capturar o h_i do trabalho.
        h[i] está definido para 0 <= i <= N, mas h[0] é posto como inf para
        ajudar nas contas.

        Similar, k[j] deve capturar o k_j do trabalho.
        k[j] está definido para 0 <= j <= M; mas k[0] é posto como inf para
        ajudar nas contas.

        É útil ter h[0] = k[0] = inf para, quando se fizer (a - b)/h[0], ter o
        resultado sendo 0, que é o desejado em vários casos.
        """

        h = np.concatenate([np.array([np.inf]), x[1:] - x[:-1]])
        k = np.concatenate([np.array([np.inf]), y[1:] - y[:-1]])

        assert all(x[i] == h[i] + x[i - 1] for i in range(1, N + 1))
        assert all(y[j] == k[j] + y[j - 1] for j in range(1, M + 1))

        """
        h_phalf[i] deve capturar o h_{i + 1/2} do artigo.
        Está definido para todo 0 <= i <= N, sendo posto como inf para
        i=0,N.

        Similar para k_phalf[j]: deve capturar o k_{j+1/2} do trabalho.
        Está definido para todo 0 <= j <= M, sendo posto como inf para k=0,M.
        """

        h_phalf = np.concatenate([(h[:-1] + h[1:]) * 0.5, np.array([np.inf])])

        k_phalf = np.concatenate([(k[:-1] + k[1:]) * 0.5, np.array([np.inf])])

        assert (
            h_phalf[0] == np.inf
            and h_phalf[N] == np.inf
            and k_phalf[0] == np.inf
            and k_phalf[M] == np.inf
        )
        assert all(h_phalf[i] == 0.5 * (h[i] + h[i + 1]) for i in range(1, N))
        assert all(k_phalf[j] == 0.5 * (k[j] + k[j + 1]) for j in range(1, M))

        self.h, self.k, self.h_phalf, self.k_phalf = h, k, h_phalf, k_phalf

        """
        A ordem trocada tem a ver com o fato de eu ter usado i para o índice
        em x e j para o índice em y, falando grosseiramente.
        """
        kk, hh = np.meshgrid(k, h)
        kk_phalf, hh_phalf = np.meshgrid(k_phalf, h_phalf)

        assert all(
            kk[i, j] == kk[0, j]
            and hh[i, j] == hh[i, 0]
            and hh_phalf[i, j] == hh_phalf[i, 0]
            and kk_phalf[i, j] == kk_phalf[0, j]
            and yy[i, j] == yy[0, j]
            and xx[i, j] == xx[i, 0]
            for i in range(0, N + 1)
            for j in range(0, M + 1)
        )

        self.hh, self.kk, self.hh_phalf, self.kk_phalf = hh, kk, hh_phalf, kk_phalf

        assert hh.shape == (N + 1, M + 1)
        assert kk.shape == (N + 1, M + 1)
        assert hh_phalf.shape == (N + 1, M + 1)
        assert kk_phalf.shape == (N + 1, M + 1)

        self.del_ij_Dx_star_ij = self.del_ab_Dx_star_ij(0, 0)
        self.del_ip1j_Dx_star_ij = self.del_ab_Dx_star_ij(1, 0)
        self.del_ij_Dy_star_ij = self.del_ab_Dy_star_ij(0, 0)
        self.del_ijp1_Dy_star_ij = self.del_ab_Dy_star_ij(0, 1)

        """
        Data for the construction of the five diagonals matrix.
        """
        d5i = np.arange(1, N)  # 1 <= i <= N-1
        d5j = np.arange(1, M)  # 1 <= j <= M-1
        d5jj, d5ii = np.meshgrid(d5j, d5i)
        d5sub = np.reshape(d5ii * (M + 1) + d5jj, (N - 1) * (M - 1))

        rspec = np.concatenate([d5sub, d5sub, d5sub, d5sub, d5sub])
        cspec = np.concatenate(
            [d5sub, d5sub - 1, d5sub + 1, d5sub - 1 - M, d5sub + 1 + M]
        )

        self.d5 = Diag5(sub=d5sub, rspec=rspec, cspec=cspec)

        self._null_bd_mask = self.const_with_nullbd(1)

    @property
    def full_shape(self):
        return (self.N + 1, self.M + 1)

    @property
    def interior_shape(self):
        return (self.N - 1, self.M - 1)

    def make_full0(self):
        return np.zeros(self.full_shape)

    @property
    def null_bd_mask(self):
        return self._null_bd_mask

    def const_with_nullbd(self, x):
        all_const = x * np.ones((self.N + 1, self.M + 1))
        all_const[:, 0] = 0
        all_const[:, -1] = 0
        all_const[0, :] = 0
        all_const[-1, :] = 0
        return all_const

    def inner_product_H(self, u, v):
        """
        <.|.>_H
        """
        return np.sum(
            u[1:-1, 1:-1]
            * np.conjugate(v[1:-1, 1:-1])
            * self.hh_phalf[1:-1, 1:-1]
            * self.kk_phalf[1:-1, 1:-1]
        )

    def norm_H(self, u):
        return np.sqrt(self.inner_product_H(u, u))

    def inner_product_pk(self, u, v):
        """
        <.|.>_{+,k}
        """
        return np.sum(
            u[1:, 1:-1]
            * np.conjugate(v[1:, 1:-1])
            * self.hh[1:, 1:-1]
            * self.kk_phalf[1:, 1:-1]
        )

    def norm_pk(self, u):
        return np.sqrt(self.inner_product_pk(u, u))

    def inner_product_hp(self, u, v):
        """
        <.|.>_{h,+}
        """
        return np.sum(
            u[1:-1, 1:]
            * np.conjugate(v[1:-1, 1:])
            * self.hh_phalf[1:-1, 1:]
            * self.kk[1:-1, 1:]
        )

    def norm_hp(self, u):
        return np.sqrt(self.inner_product_hp(u, u))

    def inner_product_p(self, ux, uy, vx, vy):
        return self.inner_product_pk(ux, vx) + self.inner_product_hp(uy, vy)

    def norm_p(self, ux, uy):
        return np.sqrt(self.inner_product_p(ux, uy, ux, uy))

    def Dx_reg(self, u):
        return Dx_reg(u, self.hh)

    def del_ab_Dx_reg_ij(self, a, b):
        return del_ab_Dx_reg_ij(a=a, b=b, N=self.N, M=self.M, hh=self.hh)

    def del_ab_Dx_reg_ip1j(self, a, b):
        return del_ab_Dx_reg_ip1j(a=a, b=b, N=self.N, M=self.M, hh=self.hh)

    def Dx_star(self, u):
        return Dx_star(u, self.hh_phalf)

    def del_ab_Dx_star_ij(self, a, b):
        return del_ab_Dx_star_ij(a=a, b=b, N=self.N, M=self.M, hh_phalf=self.hh_phalf)

    def Dy_reg(self, u):
        return Dy_reg(u, self.kk)

    def del_ab_Dy_reg_ij(self, a, b):
        return del_ab_Dy_reg_ij(a=a, b=b, N=self.N, M=self.M, kk=self.kk)

    def del_ab_Dy_reg_ijp1(self, a, b):
        return del_ab_Dy_reg_ijp1(a=a, b=b, N=self.N, M=self.M, kk=self.kk)

    def Dy_star(self, u):
        return Dy_star(u, self.kk_phalf)

    def del_ab_Dy_star_ij(self, a, b):
        return del_ab_Dy_star_ij(a=a, b=b, N=self.N, M=self.M, kk_phalf=self.kk_phalf)

    def del_ab_Mx_reg_ij(self, a, b):
        return del_ab_Mx_reg_ij(a=a, b=b, N=self.N, M=self.M)

    def del_ab_Mx_reg_ip1j(self, a, b):
        return del_ab_Mx_reg_ip1j(a=a, b=b, N=self.N, M=self.M)

    def del_ab_My_reg_ij(self, a, b):
        return del_ab_My_reg_ij(a=a, b=b, N=self.N, M=self.M)

    def del_ab_My_reg_ijp1(self, a, b):
        return del_ab_My_reg_ijp1(a=a, b=b, N=self.N, M=self.M)

    def del_ab_Id_ij(self, a, b):
        return del_ab_Id_ij(a=a, b=b, N=self.N, M=self.M)

    def del_ab_Id_ab(self, a, b):
        return del_ab_Id_ab(a=a, b=b, N=self.N, M=self.M)

    def grad_H(self, u):
        return (self.Dx_reg(u), self.Dy_reg(u))


def make_uniform_grid(N: int, M: int):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, M + 1)
    return Grid(x, y)


def avg_int(f: Callable[[np.ndarray, np.ndarray], np.ndarray], grid: Grid):
    """
    Calculates the average integral of function f(x, y) over the interior grid cells:

        [x_{i-1/2}, x_{i+1/2}] x [y_{j-1/2}, y_{j+1/2}]

    for 1 <= i <= N-1 and 1 <= j <= M-1, using 3x3 Gauss-Legendre quadrature.

    Args:
        f: A function that takes two NumPy arrays (x and y coordinates) and returns the function values at those coordinates as a NumPy array of the same shape.

    Returns:
        A NumPy array of shape (N+1, M+1) containing the average integral values for the interior cells [1:-1, 1:-1] and zeros elsewhere.
    """
    # Standard 3-point Gauss-Legendre nodes and weights for [-1, 1]
    weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])  # Weights
    nodes = np.array([-np.sqrt(3.0 / 5.0), 0, np.sqrt(3.0 / 5.0)])  # Nodes

    N = grid.N
    M = grid.M

    # ----------
    # Verification of input array shapes
    expected_shape = (N + 1, M + 1)
    assert (
        grid.xx_phalf.shape == expected_shape
    ), f"xx_phalf shape mismatch: {grid.xx_phalf.shape} != {expected_shape}"
    assert (
        grid.yy_phalf.shape == expected_shape
    ), f"yy_phalf shape mismatch: {grid.yy_phalf.shape} != {expected_shape}"
    assert (
        grid.hh_phalf.shape == expected_shape
    ), f"hh_phalf shape mismatch: {grid.hh_phalf.shape} != {expected_shape}"
    assert (
        grid.kk_phalf.shape == expected_shape
    ), f"kk_phalf shape mismatch: {grid.kk_phalf.shape} != {expected_shape}"
    # ----------

    # Initialize result array for interior points only
    # Target shape for calculations: (N-1, M-1)
    avg_int_interior = np.zeros((N - 1, M - 1))
    expected_interior_shape = (N - 1, M - 1)

    # --- Pre-calculate base points and widths for interior cells ---
    # Base x-coordinate (x_{i-1/2}) for cells i=1..N-1, j=1..M-1
    base_x = grid.xx_phalf[0 : N - 1, 1:M]
    assert (
        base_x.shape == expected_interior_shape
    ), f"base_x shape mismatch: {base_x.shape} != {expected_interior_shape}"

    # Width in x (h_{i+1/2} = x_{i+1/2} - x_{i-1/2}) for cells i=1..N-1, j=1..M-1
    width_x = grid.hh_phalf[1:N, 1:M]
    assert (
        width_x.shape == expected_interior_shape
    ), f"width_x shape mismatch: {width_x.shape} != {expected_interior_shape}"

    # Base y-coordinate (y_{j-1/2}) for cells i=1..N-1, j=1..M-1
    base_y = grid.yy_phalf[1:N, 0 : M - 1]
    assert (
        base_y.shape == expected_interior_shape
    ), f"base_y shape mismatch: {base_y.shape} != {expected_interior_shape}"

    # Width in y (k_{j+1/2} = y_{j+1/2} - y_{j-1/2}) for cells i=1..N-1, j=1..M-1
    width_y = grid.kk_phalf[1:N, 1:M]
    assert (
        width_y.shape == expected_interior_shape
    ), f"width_y shape mismatch: {width_y.shape} != {expected_interior_shape}"
    # ---------------------------------------------------------------

    # Loop over Gauss quadrature points (k_x corresponds to n[i], k_y corresponds to n[j])
    for i, node_x in enumerate(nodes):  # i is index for x-node/weight
        for j, node_y in enumerate(nodes):  # j is index for y-node/weight

            # Map nodes from [-1, 1] to the physical coordinates of each cell
            # x = base_x + (node_x + 1)/2 * width_x
            # y = base_y + (node_y + 1)/2 * width_y
            p = base_x + (node_x + 1.0) * 0.5 * width_x
            q = base_y + (node_y + 1.0) * 0.5 * width_y

            # Ensure calculated coordinates have the correct interior shape
            assert (
                p.shape == expected_interior_shape
            ), f"p shape mismatch: {p.shape} != {expected_interior_shape} for node ({i},{j})"
            assert (
                q.shape == expected_interior_shape
            ), f"q shape mismatch: {q.shape} != {expected_interior_shape} for node ({i},{j})"

            # Evaluate the function f at all quadrature points (p, q) across all interior cells
            f_values = f(p, q)

            # Ensure the function evaluation returns the correct shape
            assert (
                f_values.shape == expected_interior_shape
            ), f"Function f returned shape {f_values.shape}, expected {expected_interior_shape} for node ({i},{j})"

            # Add the weighted function values to the accumulating sum for the average integral
            # The factor 0.25 comes from the Jacobian of the transformation (Area/4)
            # divided by the Area, when calculating the average integral.
            avg_int_interior += weights[i] * weights[j] * f_values

    # Create the final result array of shape (N+1, M+1) with zeros on boundaries
    avg_int_final = np.zeros((N + 1, M + 1))
    # Final adjust: the loop calculates the SUM part of 0.25 * SUM(w*w*f).
    avg_int_final[1:-1, 1:-1] = 0.25 * avg_int_interior

    return avg_int_final


class FiveDiagonalsMatrixFactory:
    def __init__(self, N: int, M: int):
        """
        Data for the construction of the five diagonals matrix.
        """
        self.N = N
        self.M = M

        d5i = np.arange(1, N)  # 1 <= i <= N-1
        d5j = np.arange(1, M)  # 1 <= j <= M-1
        d5jj, d5ii = np.meshgrid(d5j, d5i)
        d5sub = np.reshape(d5ii * (M + 1) + d5jj, (N - 1) * (M - 1))

        rspec = np.concatenate([d5sub, d5sub, d5sub, d5sub, d5sub])
        cspec = np.concatenate(
            [d5sub, d5sub - 1, d5sub + 1, d5sub - 1 - M, d5sub + 1 + M]
        )

        self.d5 = Diag5(sub=d5sub, rspec=rspec, cspec=cspec)

    def make_matrix(
        self,
        *,
        A_ij_ij,
        A_ij_ip1j,
        A_ij_im1j,
        A_ij_ijp1,
        A_ij_ijm1,
        rem_border_entries=True,
    ):
        """
        Seja i,j quaisquer com 1 <= i <= N-1, 1 <= j <= M-1.

        A matrix A pensada em coordenadas bi-dimensionais, trabalhado no espaço
        das funções de grade nulas no bordo, faz:

        A(i,j ; i,j) = A_ij_ij(i-1,j-1)
        A(i,j ; i+1,j) = A_ij_ip1j(i-1,j-1)
        A(i,j ; i-1,j) = A_ij_im1j(i-1,j-1)
        A(i,j ; i,j+1) = A_ij_ijp1(i-1,j-1)
        A(i,j ; i,j-1) = A_ij_ijm1(i-1,j-1)

        Pensada desta forma, a matrix A é ((N+1)x(M+1)) x ((N+1)x(M+1)).
        OBS.: Nesta explicação A(i,j ; k,l) está definido para todo
        0 <= i,k <= N+1 e todo 0 <= j,l <= M+1.

        Vou fazer assumindo vetores de tamanhos (N+1)*(M+1) nulos no bordo. Fica
        mais fácil de se construir a matriz dessa forma.
        """

        N = self.N
        M = self.M

        assert A_ij_ij.shape == (N - 1, M - 1)
        assert A_ij_ip1j.shape == (N - 1, M - 1)
        assert A_ij_im1j.shape == (N - 1, M - 1)
        assert A_ij_ijp1.shape == (N - 1, M - 1)
        assert A_ij_ijm1.shape == (N - 1, M - 1)

        A_data = np.concatenate(
            [
                A_ij_ij.reshape((N - 1) * (M - 1)),
                A_ij_ijm1.reshape((N - 1) * (M - 1)),
                A_ij_ijp1.reshape((N - 1) * (M - 1)),
                A_ij_im1j.reshape((N - 1) * (M - 1)),
                A_ij_ip1j.reshape((N - 1) * (M - 1)),
            ]
        )

        """
        Talvez seja interessante calcular os índices válidos aqui e usar apenas
        eles para construir a matriz A.
        """

        A = sp.csr_array(
            (A_data, (self.d5.rspec, self.d5.cspec)),
            shape=((N + 1) * (M + 1), (N + 1) * (M + 1)),
        )

        if rem_border_entries:
            A = A[self.d5.sub].T[self.d5.sub].T
            assert A.shape == ((N - 1) * (M - 1), (N - 1) * (M - 1))

        """
        A expressão: X[l].T[l].T extrai a submatrix de X formada pelos índices
        em l.

        Por exemplo, se X for 3x3 e l = [1,2]:

        X =
            a b c
            d e f
            g h i

        X[l].T[l].T =
            e f
            h i

        Se l = [2]:

        X[l].T[l].T =
            i

        Quando faço `X = X[l].T[l].T` estou apenas jogando fora as entradas
        que correspondem tanto às saídas quanto às entradas associadas
        ao termos de bordo, ou seja, qualquer linha fora de l deve sair e
        qualquer coluna fora de l deve sair.
        """

        return A


# TODO: This doesn't need to take a grid anymore to be constructed. Remove this requirement.
class MMSCaseBase(ABC):
    """
    Defines fcp, fT, fcl, fcd, fcs in terms of exact solution, defined in a sub-class.
    """

    def __init__(self, grid: Grid, model: DefaultModel01):
        """
        Remark: Ideally, MMS Cases would be defined in a spatial/time-discretization independent way. Therefore, it is conceptually strange for MMSCaseBase to need a grid. However, all MMS Cases are meant, in this current state of the program, to be used under a particular grid configuration, which must be provided. "Difficult" forcing terms deduction might not be doable in a fully general way without resorting to the contextually known spatial discretization.
        """
        self._model = model
        self._grid = grid
        self._xx = grid.xx
        self._yy = grid.yy

    @property
    def grid(self):
        return self._grid

    @property
    def model(self):
        return self._model

    @abstractmethod
    def dt_cp(self, t, xx, yy):
        pass

    @abstractmethod
    def dt_T(self, t, xx, yy):
        pass

    @abstractmethod
    def dt_cl(self, t, xx, yy):
        pass

    @abstractmethod
    def dt_cd(self, t, xx, yy):
        pass

    @abstractmethod
    def dt_cs(self, t, xx, yy):
        pass

    @abstractmethod
    def lap_T(self, t, xx, yy):
        pass

    @abstractmethod
    def lap_cl(self, t, xx, yy):
        pass

    @abstractmethod
    def lap_cd(self, t, xx, yy):
        pass

    @abstractmethod
    def dx_cp(self, t, xx, yy):
        pass

    @abstractmethod
    def dy_cp(self, t, xx, yy):
        pass

    @abstractmethod
    def dx_T(self, t, xx, yy):
        pass

    @abstractmethod
    def dy_T(self, t, xx, yy):
        pass

    @abstractmethod
    def dx_cl(self, t, xx, yy):
        pass

    @abstractmethod
    def dy_cl(self, t, xx, yy):
        pass

    @abstractmethod
    def dx_cd(self, t, xx, yy):
        pass

    @abstractmethod
    def dy_cd(self, t, xx, yy):
        pass

    @abstractmethod
    def cp(self, t, xx, yy):
        pass

    @abstractmethod
    def cs(self, t, xx, yy):
        pass

    @abstractmethod
    def T(self, t, xx, yy):
        pass

    @abstractmethod
    def cl(self, t, xx, yy):
        pass

    @abstractmethod
    def cd(self, t, xx, yy):
        pass


class ForcingTermsBase(ABC):
    @abstractmethod
    def fcp(self, t, xx, yy) -> np.ndarray:
        pass

    @abstractmethod
    def fT(self, t, xx, yy) -> np.ndarray:
        pass

    @abstractmethod
    def fcl(self, t, xx, yy) -> np.ndarray:
        pass

    @abstractmethod
    def fcd(self, t, xx, yy) -> np.ndarray:
        pass

    @abstractmethod
    def fcs(self, t, xx, yy) -> np.ndarray:
        pass

    def asdict(self) -> Dict[str, Callable[[float], np.ndarray]]:
        return {
            "fcp": self.fcp,
            "fT": self.fT,
            "fcl": self.fcl,
            "fcd": self.fcd,
            "fcs": self.fcs,
        }


class NoForcingTerms(ForcingTermsBase):
    def __init__(self, grid: Grid):
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


class ForcingTermsFromDict(ForcingTermsBase):
    def __init__(self, forcing_terms_dict: Dict):
        self._forcing_terms_dict = forcing_terms_dict

    def fcp(self, t, xx, yy) -> np.ndarray:
        return self._forcing_terms_dict["fcp"](t, xx, yy)

    def fT(self, t, xx, yy) -> np.ndarray:
        return self._forcing_terms_dict["fT"](t, xx, yy)

    def fcl(self, t, xx, yy) -> np.ndarray:
        return self._forcing_terms_dict["fcl"](t, xx, yy)

    def fcd(self, t, xx, yy) -> np.ndarray:
        return self._forcing_terms_dict["fcd"](t, xx, yy)

    def fcs(self, t, xx, yy) -> np.ndarray:
        return self._forcing_terms_dict["fcs"](t, xx, yy)


TimeSteppingStrategy = Literal["forward", "center", "backward"]


def pack_analytical_txy_with_o2fdm_derivatives(
    fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    *,
    default_eps: float = 1e-6,
    time_stepping: TimeSteppingStrategy = "center",
) -> Callable:
    """
    Enhances a function to support various derivative orders and differential operators using 2nd order finite differences.

    Args:
        fn: Original function taking (t, x, y) as arguments
        default_eps: Default step size for finite differences (default: 1e-6)
        time_stepping: Strategy for time derivatives ("forward", "center", or "backward")

    Returns:
        Enhanced function with additional derivative and operator capabilities
    """

    def dt1_forward(t: float, x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
        return (-3 * fn(t, x, y) + 4 * fn(t + eps, x, y) - fn(t + 2 * eps, x, y)) / (
            2 * eps
        )

    def dt1_center(t: float, x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
        return (fn(t + eps, x, y) - fn(t - eps, x, y)) / (2 * eps)

    def dt1_backward(t: float, x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
        return (3 * fn(t, x, y) - 4 * fn(t - eps, x, y) + fn(t - 2 * eps, x, y)) / (
            2 * eps
        )

    def dt2_center(t: float, x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
        return (fn(t + eps, x, y) - 2 * fn(t, x, y) + fn(t - eps, x, y)) / (eps * eps)

    def dt2_forward(t: float, x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
        # 2nd order accurate forward difference for 2nd derivative
        return (
            2 * fn(t, x, y)
            - 5 * fn(t + eps, x, y)
            + 4 * fn(t + 2 * eps, x, y)
            - fn(t + 3 * eps, x, y)
        ) / (eps * eps)

    def dt2_backward(t: float, x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
        # 2nd order accurate backward difference for 2nd derivative
        return (
            2 * fn(t, x, y)
            - 5 * fn(t - eps, x, y)
            + 4 * fn(t - 2 * eps, x, y)
            - fn(t - 3 * eps, x, y)
        ) / (eps * eps)

    # Select time derivative functions based on strategy
    if time_stepping == "forward":
        dt1_fn = dt1_forward
        dt2_fn = dt2_forward
    elif time_stepping == "center":
        dt1_fn = dt1_center
        dt2_fn = dt2_center
    elif time_stepping == "backward":
        dt1_fn = dt1_backward
        dt2_fn = dt2_backward
    else:
        raise ValueError("Invalid time stepping strategy")

    def enhanced_fn(
        t: float,
        x: np.ndarray,
        y: np.ndarray,
        *,
        d: Tuple[int, int, int] = (0, 0, 0),
        op: Optional[str] = None,
        small_eps: float = None,
    ) -> np.ndarray:
        eps = small_eps or default_eps

        # Handle special operators first
        if op is not None:
            op = op.lower()
            if op in ["laplacian", "lap"]:
                return (
                    fn(t, x + eps, y)
                    + fn(t, x - eps, y)
                    + fn(t, x, y + eps)
                    + fn(t, x, y - eps)
                    - 4 * fn(t, x, y)
                ) / (eps * eps)

            else:
                raise ValueError(f"Unknown operator: {op}. Use 'laplacian'/'lap'")

        # Regular derivatives processing
        dt, dx, dy = d

        # Assert that individual derivatives are 0, 1, or 2
        if not all(d in [0, 1, 2] for d in [dt, dx, dy]):
            raise ValueError("Individual derivatives must be 0, 1, or 2")

        # Assert that combined order is 0, 1, or 2
        total_order = dt + dx + dy
        if total_order > 2:
            raise ValueError("Combined derivative order must be 0, 1, or 2")

        # Handle t derivatives using selected strategy
        if dt == 1:
            return dt1_fn(t, x, y, eps)
        elif dt == 2:
            return dt2_fn(t, x, y, eps)

        # Handle mixed xy derivatives first
        if dx == 1 and dy == 1:
            return (
                fn(t, x + eps, y + eps)
                - fn(t, x + eps, y - eps)
                - fn(t, x - eps, y + eps)
                + fn(t, x - eps, y - eps)
            ) / (4 * eps * eps)

        # Handle x derivatives
        if dx == 1:
            return (fn(t, x + eps, y) - fn(t, x - eps, y)) / (2 * eps)
        elif dx == 2:
            return (fn(t, x + eps, y) - 2 * fn(t, x, y) + fn(t, x - eps, y)) / (
                eps * eps
            )

        # Handle y derivatives
        if dy == 1:
            return (fn(t, x, y + eps) - fn(t, x, y - eps)) / (2 * eps)
        elif dy == 2:
            return (fn(t, x, y + eps) - 2 * fn(t, x, y) + fn(t, x, y - eps)) / (
                eps * eps
            )

        # If no derivatives or operators, return original function
        return fn(t, x, y)

    return enhanced_fn


# NOTE: Only use this when MMSCaseSymbolic isn't applicable.
class MMSCaseFromAnalytic(MMSCaseBase):
    """
    Method of Manufactured Solutions (MMS) case derived from analytically defined (i.e. by standard formulas) mathematical functions in regular Python functions.

    This class is an alternative to `MMSCaseSymbolic` for situations where the exact solutions for cp, T, cl, cd, and cs are provided as Python callable functions `fn(t, xx, yy)` rather than SymPy expressions.

    It uses the `pack_analytical_txy_with_o2fdm_derivatives` utility to wrap these base analytical functions. This wrapper provides numerical approximations for the required time derivatives (dt), spatial derivatives (dx, dy), and Laplacians (lap) using second-order finite difference methods.

    The primary purpose is to allow MMS setup without requiring symbolic manipulation, relying instead on numerical differentiation of the provided analytical solution functions.

    Attributes:
        cp_ex (Callable): Wrapped function for cp, providing derivatives.
        T_ex (Callable): Wrapped function for T, providing derivatives.
        cl_ex (Callable): Wrapped function for cl, providing derivatives.
        cd_ex (Callable): Wrapped function for cd, providing derivatives.
        cs_ex (Callable): Wrapped function for cs, providing derivatives.
    """

    def __init__(
        self,
        model: DefaultModel01,
        *,
        grid: Grid,
        cp_base,
        T_base,
        cl_base,
        cd_base,
        cs_base,
    ):
        """
        Initializes the MMS case with Python functions for each variable.

        Args:
            model: An instance of `DefaultModel01` (or a compatible model)
                   containing physical constants.
            grid: A `Grid` object providing the `xx` and `yy` coordinate
                  arrays for evaluating the solutions.
            cp_base: A callable function `f(t, xx, yy)` representing the exact
                     analytical solution for cp.
            T_base: A callable function `f(t, xx, yy)` for T.
            cl_base: A callable function `f(t, xx, yy)` for cl.
            cd_base: A callable function `f(t, xx, yy)` for cd.
            cs_base: A callable function `f(t, xx, yy)` for cs.
        """
        super().__init__(grid, model)

        self.cp_ex = pack_analytical_txy_with_o2fdm_derivatives(cp_base)
        self.T_ex = pack_analytical_txy_with_o2fdm_derivatives(T_base)
        self.cl_ex = pack_analytical_txy_with_o2fdm_derivatives(cl_base)
        self.cd_ex = pack_analytical_txy_with_o2fdm_derivatives(cd_base)
        self.cs_ex = pack_analytical_txy_with_o2fdm_derivatives(cs_base)

    def cp(self, t, xx, yy):
        return self.cp_ex(t, xx, yy)

    def cs(self, t, xx, yy):
        return self.cs_ex(t, xx, yy)

    def T(self, t, xx, yy):
        return self.T_ex(t, xx, yy)

    def cl(self, t, xx, yy):
        return self.cl_ex(t, xx, yy)

    def cd(self, t, xx, yy):
        return self.cd_ex(t, xx, yy)

    def dt_cp(self, t, xx, yy):
        return self.cp_ex(t, xx, yy, d=(1, 0, 0))

    def dt_cs(self, t, xx, yy):
        return self.cs_ex(t, xx, yy, d=(1, 0, 0))

    def dt_T(self, t, xx, yy):
        return self.T_ex(t, xx, yy, d=(1, 0, 0))

    def dt_cl(self, t, xx, yy):
        return self.cl_ex(t, xx, yy, d=(1, 0, 0))

    def dt_cd(self, t, xx, yy):
        return self.cd_ex(t, xx, yy, d=(1, 0, 0))

    def lap_T(self, t, xx, yy):
        return self.T_ex(t, xx, yy, d=(0, 2, 0)) + self.T_ex(t, xx, yy, d=(0, 0, 2))

    def lap_cl(self, t, xx, yy):
        return self.cl_ex(t, xx, yy, d=(0, 2, 0)) + self.cl_ex(t, xx, yy, d=(0, 0, 2))

    def lap_cd(self, t, xx, yy):
        return self.cd_ex(t, xx, yy, d=(0, 2, 0)) + self.cd_ex(t, xx, yy, d=(0, 0, 2))

    def dx_cp(self, t, xx, yy):
        return self.cp_ex(t, xx, yy, d=(0, 1, 0))

    def dy_cp(self, t, xx, yy):
        return self.cp_ex(t, xx, yy, d=(0, 0, 1))

    def dx_cs(self, t, xx, yy):
        return self.cs_ex(t, xx, yy, d=(0, 1, 0))

    def dy_cs(self, t, xx, yy):
        return self.cs_ex(t, xx, yy, d=(0, 0, 1))

    def dx_T(self, t, xx, yy):
        return self.T_ex(t, xx, yy, d=(0, 1, 0))

    def dy_T(self, t, xx, yy):
        return self.T_ex(t, xx, yy, d=(0, 0, 1))

    def dx_cl(self, t, xx, yy):
        return self.cl_ex(t, xx, yy, d=(0, 1, 0))

    def dy_cl(self, t, xx, yy):
        return self.cl_ex(t, xx, yy, d=(0, 0, 1))

    def dx_cd(self, t, xx, yy):
        return self.cd_ex(t, xx, yy, d=(0, 1, 0))

    def dy_cd(self, t, xx, yy):
        return self.cd_ex(t, xx, yy, d=(0, 0, 1))


# --- Symbolically define exact solutions ---

# It's good practice to define the symbolic variables once globally
# if they are always going to be the same ('t', 'x', 'y').
# Users creating symbolic expressions should import and use these.
# 0 < t, 0 <= x,y <= 1.
t_sym, x_sym, y_sym = sympy.symbols("t x y", negative=False, real=True)


def _create_shape_adjusting_wrapper(raw_lambdified_func: Callable) -> Callable:
    """
    Internal helper to create a wrapper around a lambdified function.

    The wrapper ensures the output has the same form as the input. It also makes sure inputs are of the same shape. Output is forced to be of that same shape.

    This assumes a function `raw_lambdified_func` that computes some z = f(x,y). As arrays, `z`, `x`, `y` must be of same shape, always.
    """

    def wrapped_func(t_num: float, x_num: np.ndarray, y_num: np.ndarray) -> np.ndarray:
        """
        Wraps a lambdified function to ensure proper shape handling of inputs and outputs.

        This function serves as a wrapper around a raw lambdified function, ensuring consistent shape handling. It evaluates the raw function with the given inputs and reshapes the output according to the shape of the spatial inputs.

        Parameters:
        -----------
        t_num : number
            A scalar time value.
        x_num : array_like
            Spatial x-coordinates, can be scalar or array.
        y_num : array_like
            Spatial y-coordinates, must have the same shape as x_num.

        Returns:
        --------
        numpy.ndarray
            The result of the lambdified function, reshaped to match the input spatial coordinates shape. If the raw result is scalar, it is broadcast to match the target shape. dtype of resulting array is always np.float64.

        Raises:
        -------
        AssertionError
            If t_num is not a number, if x_num and y_num have different shapes,
            or if the flattened result size doesn't match the product of the target shape.
        """
        # Evaluate the raw lambdified function
        assert isinstance(t_num, numbers.Number)
        raw_result = raw_lambdified_func(t_num, x_num, y_num)

        x_shape = np.shape(x_num)
        y_shape = np.shape(y_num)

        assert x_shape == y_shape
        target_shape = x_shape

        raw_flat = np.asarray(raw_result).flatten()
        n_raw_flat = len(raw_flat)
        is_result_scalar = n_raw_flat == 1

        if is_result_scalar:
            raw_result = raw_flat[0]
            return np.full(target_shape, raw_result, dtype=np.float64)
        else:
            assert n_raw_flat == np.prod(target_shape)
            return np.reshape(raw_result, target_shape).astype(np.float64)

    return wrapped_func


def pack_symbolic_txy_with_derivatives(
    *,
    base_expr: sympy.Expr,
    t_var: sympy.Symbol = t_sym,
    x_var: sympy.Symbol = x_sym,
    y_var: sympy.Symbol = y_sym,
) -> Dict[str, Callable]:
    """
    Calculates symbolic derivatives and the Laplacian of a base symbolic
    expression and returns a dictionary of lambdified functions that consistently
    return NumPy arrays matching the shape of input coordinate arrays.

    Args:
        base_expr: The core sympy expression depending on t_var, x_var, y_var.
        t_var: The sympy symbol used for time in base_expr (default: t_sym).
        x_var: The sympy symbol used for the x-coordinate (default: x_sym).
        y_var: The sympy symbol used for the y-coordinate (default: y_sym).

    Returns:
        A dictionary mapping descriptor strings to callable functions ('base',
        'dt', 'dx', 'dy', 'dxx', 'dyy', 'lap'). Each function takes numerical
        inputs (t: float, x: np.ndarray, y: np.ndarray) and returns a NumPy array
        matching the shape of x and y arrays, handling promotion of scalar results.
    """
    func_dict_raw = {}

    # Define derivatives symbolically
    dt_expr = sympy.diff(base_expr, t_var)
    dtt_expr = sympy.diff(dt_expr, t_var)
    dx_expr = sympy.diff(base_expr, x_var)
    dy_expr = sympy.diff(base_expr, y_var)
    dxx_expr = sympy.diff(dx_expr, x_var)
    dyy_expr = sympy.diff(dy_expr, y_var)
    lap_expr = dxx_expr + dyy_expr

    # Custom modules for lambdify to handle DiracDelta
    custom_modules = [{'DiracDelta': lambda arg: np.where(abs(arg) < 1e-13, 1.0, 0.0)}, "numpy"]

    # Create RAW lambdified functions
    eval_vars = [t_var, x_var, y_var]
    func_dict_raw["base"] = sympy.lambdify(eval_vars, base_expr, modules=custom_modules)
    func_dict_raw["dt"] = sympy.lambdify(eval_vars, dt_expr, modules=custom_modules)
    func_dict_raw["dtt"] = sympy.lambdify(eval_vars, dtt_expr, modules=custom_modules)
    func_dict_raw["dx"] = sympy.lambdify(eval_vars, dx_expr, modules=custom_modules)
    func_dict_raw["dy"] = sympy.lambdify(eval_vars, dy_expr, modules=custom_modules)
    func_dict_raw["dxx"] = sympy.lambdify(eval_vars, dxx_expr, modules=custom_modules)
    func_dict_raw["dyy"] = sympy.lambdify(eval_vars, dyy_expr, modules=custom_modules)
    func_dict_raw["lap"] = sympy.lambdify(eval_vars, lap_expr, modules=custom_modules)

    # Wrap each raw function to handle shape adjustment
    func_dict_wrapped = {}
    for name, raw_func in func_dict_raw.items():
        func_dict_wrapped[name] = _create_shape_adjusting_wrapper(raw_func)

    return func_dict_wrapped


class MMSCaseSymbolic(MMSCaseBase):
    """
    Implementation of MMSCaseBase using symbolic math (SymPy) to define
    the exact solution and automatically derive numerical derivatives required
    by the base class forcing term calculations.
    """

    def __init__(
        self,
        *,
        grid: Grid,
        model: DefaultModel01,
        cp_sym_expr: sympy.Expr,
        T_sym_expr: sympy.Expr,
        cl_sym_expr: sympy.Expr,
        cd_sym_expr: sympy.Expr,
        cs_sym_expr: sympy.Expr,
        t_var: sympy.Symbol = t_sym,  # Allow specifying symbols used
        x_var: sympy.Symbol = x_sym,
        y_var: sympy.Symbol = y_sym,
    ):
        """
        Initializes the pack with symbolic expressions for the exact solution.
        It derives and lambdifies numerical functions for all required abstract
        methods of MMSCaseBase. Forcing terms are calculated via the base
        class methods.

        Args:
            grid: The Grid object providing numerical evaluation coordinates.
            model: The DefaultModel01 instance with physical constants.
            cp_sym_expr: Sympy expression for cp(t, x, y).
            T_sym_expr: Sympy expression for T(t, x, y).
            cl_sym_expr: Sympy expression for cl(t, x, y).
            cd_sym_expr: Sympy expression for cd(t, x, y).
            cs_sym_expr: Sympy expression for cs(t, x, y).
            t_var: The sympy symbol for time used in the expressions.
            x_var: The sympy symbol for x used in the expressions.
            y_var: The sympy symbol for y used in the expressions.
        """
        # Initialize the base class, passing the model
        super().__init__(grid, model)

        # Store the model instance directly if needed (super already does)
        # self.model = model

        # Create dictionaries of callable NUMERICAL functions for each variable
        # using the symbolic packager utility

        pack_args = {"t_var": t_var, "x_var": x_var, "y_var": y_var}
        self._cp_pack = pack_symbolic_txy_with_derivatives(
            base_expr=cp_sym_expr, **pack_args
        )
        self._T_pack = pack_symbolic_txy_with_derivatives(
            base_expr=T_sym_expr, **pack_args
        )
        self._cl_pack = pack_symbolic_txy_with_derivatives(
            base_expr=cl_sym_expr, **pack_args
        )
        self._cd_pack = pack_symbolic_txy_with_derivatives(
            base_expr=cd_sym_expr, **pack_args
        )
        self._cs_pack = pack_symbolic_txy_with_derivatives(
            base_expr=cs_sym_expr, **pack_args
        )

        # No symbolic forcing term calculation needed here anymore.
        # The parent class forcing methods (fcp, fT, etc.) will be called,
        # and they will internally call the abstract methods implemented below.

    @property
    def cp_pack(self) -> Dict[str, Callable]:
        return self._cp_pack

    @property
    def T_pack(self) -> Dict[str, Callable]:
        return self._T_pack

    @property
    def cl_pack(self) -> Dict[str, Callable]:
        return self._cl_pack

    @property
    def cd_pack(self) -> Dict[str, Callable]:
        return self._cd_pack

    @property
    def cs_pack(self) -> Dict[str, Callable]:
        return self._cs_pack

    # --- Implement ALL abstract methods from MMSCaseBase ---
    # --- These methods return NUMERICAL NumPy arrays ---

    def cp(self, t, xx, yy):
        return self.cp_pack["base"](t, xx, yy)

    def cs(self, t, xx, yy):
        return self.cs_pack["base"](t, xx, yy)

    def T(self, t, xx, yy):
        return self.T_pack["base"](t, xx, yy)

    def cl(self, t, xx, yy):
        return self.cl_pack["base"](t, xx, yy)

    def cd(self, t, xx, yy):
        return self.cd_pack["base"](t, xx, yy)

    def dt_cp(self, t, xx, yy):
        return self.cp_pack["dt"](t, xx, yy)

    def dt_T(self, t, xx, yy):
        return self.T_pack["dt"](t, xx, yy)

    def dt_cl(self, t, xx, yy):
        return self.cl_pack["dt"](t, xx, yy)

    def dt_cd(self, t, xx, yy):
        return self.cd_pack["dt"](t, xx, yy)

    def dt_cs(self, t, xx, yy):
        return self.cs_pack["dt"](t, xx, yy)

    def dtt_cp(self, t, xx, yy):
        return self.cp_pack["dtt"](t, xx, yy)

    def dtt_T(self, t, xx, yy):
        return self.T_pack["dtt"](t, xx, yy)

    def dtt_cl(self, t, xx, yy):
        return self.cl_pack["dtt"](t, xx, yy)

    def dtt_cd(self, t, xx, yy):
        return self.cd_pack["dtt"](t, xx, yy)

    def dtt_cs(self, t, xx, yy):
        return self.cs_pack["dtt"](t, xx, yy)

    def lap_T(self, t, xx, yy):
        return self.T_pack["lap"](t, xx, yy)

    def lap_cl(self, t, xx, yy):
        return self.cl_pack["lap"](t, xx, yy)

    def lap_cd(self, t, xx, yy):
        return self.cd_pack["lap"](t, xx, yy)

    def dx_cp(self, t, xx, yy):
        return self.cp_pack["dx"](t, xx, yy)

    def dy_cp(self, t, xx, yy):
        return self.cp_pack["dy"](t, xx, yy)

    def dx_T(self, t, xx, yy):
        return self.T_pack["dx"](t, xx, yy)

    def dy_T(self, t, xx, yy):
        return self.T_pack["dy"](t, xx, yy)

    def dx_cl(self, t, xx, yy):
        return self.cl_pack["dx"](t, xx, yy)

    def dy_cl(self, t, xx, yy):
        return self.cl_pack["dy"](t, xx, yy)

    def dx_cd(self, t, xx, yy):
        return self.cd_pack["dx"](t, xx, yy)

    def dy_cd(self, t, xx, yy):
        return self.cd_pack["dy"](t, xx, yy)

    def dx_cs(self, t, xx, yy):
        return self.cs_pack["dx"](t, xx, yy)

    def dy_cs(self, t, xx, yy):
        return self.cs_pack["dy"](t, xx, yy)

    def dxx_cp(self, t, xx, yy):
        return self.cp_pack["dxx"](t, xx, yy)

    def dyy_cp(self, t, xx, yy):
        return self.cp_pack["dyy"](t, xx, yy)

    def dxx_T(self, t, xx, yy):
        return self.T_pack["dxx"](t, xx, yy)

    def dyy_T(self, t, xx, yy):
        return self.T_pack["dyy"](t, xx, yy)

    def dxx_cl(self, t, xx, yy):
        return self.cl_pack["dxx"](t, xx, yy)

    def dyy_cl(self, t, xx, yy):
        return self.cl_pack["dyy"](t, xx, yy)

    def dxx_cd(self, t, xx, yy):
        return self.cd_pack["dxx"](t, xx, yy)

    def dyy_cd(self, t, xx, yy):
        return self.cd_pack["dyy"](t, xx, yy)

    def dxx_cs(self, t, xx, yy):
        return self.cs_pack["dxx"](t, xx, yy)

    def dyy_cs(self, t, xx, yy):
        return self.cs_pack["dyy"](t, xx, yy)


def assert_del_ab(a, b):
    """
    This function checks if the arguments a and b are valid for a axis-aligned 1-distance neighbor specification. More precisely: It asserts that a and b are in the set {-1, 0, 1} and that at least one of them is 0.
    """
    assert a in [-1, 0, 1]
    assert b in [-1, 0, 1]
    assert a == 0 or b == 0


def Mx_reg(u):
    """
    M_{x}u
    """
    Mxu = np.zeros_like(u)
    Mxu[1:, :] = 0.5 * (u[1:, :] + u[:-1, :])
    return Mxu


def My_reg(u):
    """
    M_{y}u
    """
    Myu = np.zeros_like(u)
    Myu[:, 1:] = 0.5 * (u[:, 1:] + u[:, :-1])
    return Myu


def Dx_reg(u, hh):
    """
    D_{-x}u
    """
    Dmxu = np.zeros_like(u)
    Dmxu[1:, :] = (u[1:, :] - u[:-1, :]) / hh[1:, :]
    return Dmxu


def Dx_star(u, hh_phalf):
    """
    D*_{x}u
    """
    Dxu = np.zeros_like(u)
    Dxu[1:-1, :] = (u[2:, :] - u[1:-1, :]) / hh_phalf[1:-1, :]
    return Dxu


def Dy_reg(u, kk):
    """
    D_{-y}u
    """
    Dmyu = np.zeros_like(u)
    Dmyu[:, 1:] = (u[:, 1:] - u[:, :-1]) / kk[:, 1:]
    return Dmyu


def Dy_star(u, kk_phalf):
    """
    D*_{y}u
    """
    Dyu = np.zeros_like(u)
    Dyu[:, 1:-1] = (u[:, 2:] - u[:, 1:-1]) / kk_phalf[:, 1:-1]
    return Dyu


def grid0_fn_ab(u, a, b):
    """
    This function returns a grid function v, such that v[i,j] = u[i+a,j+b].

    The arguments a and b must be in the set {-1, 0, 1} and at least one of them must be 0.

    More precisely:

    v[i,j] = u[i+a,j+b]

    (a,b) in {(-1,0), (1,0), (0,-1), (0,1)}
    """

    assert_del_ab(a, b)

    v = np.zeros_like(u)

    if a == 0 and b == 0:
        v = np.copy(u)
    elif a == 1:
        v[:-1, :] = u[1:, :]
    elif a == -1:
        v[1:, :] = u[:-1, :]
    elif b == 1:
        v[:, :-1] = u[:, 1:]
    elif b == -1:
        v[:, 1:] = u[:, :-1]

    return v


def del_ab_Dx_reg_ij(*, N, M, a, b, hh):
    """
    This function returns the derivative of D_{-x}u with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} Dmx_ij
    # Dmx_ij = (u_{ij} - u_{i-1,j})/h_i
    # del_{i+a, j+b} Dmx_ij = (del_{i+a, j+b} u_{ij} - del_{i+a, j+b} u_{i-1,j})/h_i
    #     = (dirac_{i+a, j+b; ij} - dirac_{i+a, j+b; i-1,j})/h_i
    #     = (dirac_{a, b; 0,0} - dirac_{a, b; -1,0})/h_i
    """
    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ij = int(a == 0 and b == 0)
    dirac_ab_im1j = int(a == -1 and b == 0)

    deriv[1:-1, 1:-1] = (dirac_ab_ij - dirac_ab_im1j) / hh[1:-1, 1:-1]
    return deriv


def del_ab_Dx_reg_ip1j(*, N, M, a, b, hh):
    """
    This function returns the derivative of D_{-x}u_{i+1,j} with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} Dmx_{i+1,j}
    # Dmx_{i+1,j} = (u_{i+1,j} - u_{ij})/h_{i+1}
    # del_{i+a, j+b} Dmx_{i+1,j} = (del_{i+a, j+b} u_{i+1,j} - del_{i+a, j+b} u_{ij})/h_{i+1}
    #     = (dirac_{i+a, j+b; i+1,j} - dirac_{i+a, j+b; i, j})/h_{i+1}
    #     = (dirac_{a, b; 1,0} - dirac_{a, b; 0,0})/h_{i+1}
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ij = int(a == 0 and b == 0)
    dirac_ab_ip1j = int(a == 1 and b == 0)

    deriv[1:-1, 1:-1] = (dirac_ab_ip1j - dirac_ab_ij) / hh[2:, 1:-1]
    return deriv


def del_ab_Dy_reg_ij(*, N, M, a, b, kk):
    """
    This function returns the derivative of D_{-y}u with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} Dmy_ij
    # Dmy_ij = (u_{ij} - u_{i,j-1})/k_j
    # del_{i+a, j+b} Dmy_ij = (del_{i+a, j+b} u_{ij} - del_{i+a, j+b} u_{i,j-1})/k_j
    #     = (dirac_{i+a, j+b; ij} - dirac_{i+a, j+b; i,j-1})/k_j
    #     = (dirac_{a, b; 0,0} - dirac_{a, b; 0,-1})/k_j
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ij = int(a == 0 and b == 0)
    dirac_ab_ijm1 = int(a == 0 and b == -1)

    deriv[1:-1, 1:-1] = (dirac_ab_ij - dirac_ab_ijm1) / kk[1:-1, 1:-1]
    return deriv


def del_ab_Dy_reg_ijp1(*, N, M, a, b, kk):
    """
    This function returns the derivative of D_{-y}u_{i,j+1} with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} Dmy_{i,j+1}
    # Dmy_{i,j+1} = (u_{i,j+1} - u_{i,j})/k_{j+1}
    # del_{i+a, j+b} Dmy_{i,j+1} = (del_{i+a, j+b} u_{i,j+1} - del_{i+a, j+b} u_{ij})/k_{j+1}
    #     = (dirac_{i+a, j+b; i,j+1} - dirac_{i+a, j+b; i,j})/k_{j+1}
    #     = (dirac_{a, b; 0,1} - dirac_{a, b; 0,0})/k_{j+1}
    """
    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ij = int(a == 0 and b == 0)
    dirac_ab_ijp1 = int(a == 0 and b == 1)

    deriv[1:-1, 1:-1] = (dirac_ab_ijp1 - dirac_ab_ij) / kk[1:-1, 2:]
    return deriv


def del_ab_Dx_star_ij(*, N, M, a, b, hh_phalf):
    """
    This function returns the derivative of D*_{x}u with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} Dx*_ij
    # Dx*_ij = (u_{i+1,j} - u_{ij})/h_{i+1/2}
    # del_{i+a, j+b} Dx*_ij = (del_{i+a, j+b} u_{i+1,j} - del_{i+a, j+b} u_{ij})/h_{i+1/2}
    #     = (dirac_{i+a, j+b; i+1,j} - dirac_{i+a, j+b; i,j})/h_{i+1/2}
    #     = (dirac_{a, b; 1,0} - dirac_{a, b; 0,0})/h_{i+1/2}
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ip1j = int(a == 1 and b == 0)
    dirac_ab_ij = int(a == 0 and b == 0)

    deriv[1:-1, 1:-1] = (dirac_ab_ip1j - dirac_ab_ij) / hh_phalf[1:-1, 1:-1]
    return deriv


def del_ab_Dy_star_ij(*, N, M, a, b, kk_phalf):
    """
    This function returns the derivative of Dy*u with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} Dy*_ij
    # Dy*_ij = (u_{i,j+1} - u_{ij})/k_{j+1/2}
    # del_{i+a, j+b} Dy*_ij = (del_{i+a, j+b} u_{i,j+1} - del_{i+a, j+b} u_{ij})/k_{j+1/2}
    #     = (dirac_{i+a, j+b; i,j+1} - dirac_{i+a, j+b; i,j})/k_{j+1/2}
    #     = (dirac_{a, b; 0,1} - dirac_{a, b; 0,0})/k_{j+1/2}
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ijp1 = int(a == 0 and b == 1)
    dirac_ab_ij = int(a == 0 and b == 0)

    deriv[1:-1, 1:-1] = (dirac_ab_ijp1 - dirac_ab_ij) / kk_phalf[1:-1, 1:-1]
    return deriv


def del_ab_Mx_reg_ij(*, N, M, a, b):
    """
    This function returns the derivative of M_{x}u with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} (Mx_reg)_ij
    # (Mx_reg)_ij = (u_{ij} + u_{i-1,j})/2
    # del_{i+a, j+b} (Mx_reg)_ij = (del_{i+a, j+b} u_{ij} + del_{i+a, j+b} u_{i-1,j})/2
    #     = (dirac_{i+a, j+b; i, j} + dirac_{i+a, j+b; i-1,j})/2
    #     = (dirac_{a, b; 0,0} + dirac_{a, b; -1,0})/2
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ij = int(a == 0 and b == 0)
    dirac_ab_im1j = int(a == -1 and b == 0)

    deriv[1:-1, 1:-1] = (dirac_ab_ij + dirac_ab_im1j) * 0.5
    return deriv


def del_ab_Mx_reg_ip1j(*, N, M, a, b):
    """
    This function returns the derivative of M_{x}u_{i+1,j} with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} (Mx_reg)_{i+1,j}
    # (Mx_reg)_{i+1,j} = (u_{i+1,j} + u_{ij})/2
    # del_{i+a, j+b} (Mx_reg)_{i+1,j} = (del_{i+a, j+b} u_{i+1,j} + del_{i+a, j+b} u_{ij})/2
    #     = (dirac_{i+a, j+b; i+1, j} + dirac_{i+a, j+b; i,j})/2
    #     = (dirac_{a, b; 1,0} + dirac_{a, b; 0,0})/2
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ip1j = int(a == 1 and b == 0)
    dirac_ab_ij = int(a == 0 and b == 0)

    deriv[1:-1, 1:-1] = (dirac_ab_ip1j + dirac_ab_ij) * 0.5
    return deriv


def del_ab_My_reg_ij(*, N, M, a, b):
    """
    This function returns the derivative of M_{y}u with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} (My_reg)_ij
    # (My_reg)_ij = (u_{ij} + u_{i,j-1})/2
    # del_{i+a, j+b} (My_reg)_ij = (del_{i+a, j+b} u_{ij} + del_{i+a, j+b} u_{i,j-1})/2
    #     = (dirac_{i+a, j+b; i, j} + dirac_{i+a, j+b; i,j-1})/2
    #     = (dirac_{a, b; 0,0} + dirac_{a, b; 0,-1})/2
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ij = int(a == 0 and b == 0)
    dirac_ab_ijm1 = int(a == 0 and b == -1)

    deriv[1:-1, 1:-1] = (dirac_ab_ij + dirac_ab_ijm1) * 0.5
    return deriv


def del_ab_My_reg_ijp1(*, N, M, a, b):
    """
    This function returns the derivative of M_{y}u_{i,j+1} with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} (My_reg)_{i,j+1}
    # (My_reg)_{i,j+1} = (u_{i,j+1} + u_{ij})/2
    # del_{i+a, j+b} (My_reg)_{i,j+1} = (del_{i+a, j+b} u_{i,j+1} + del_{i+a, j+b} u_{ij})/2
    #     = (dirac_{i+a, j+b; i, j+1} + dirac_{i+a, j+b; i,j})/2
    #     = (dirac_{a, b; 0,1} + dirac_{a, b; 0,0})/2
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))

    dirac_ab_ijp1 = int(a == 0 and b == 1)
    dirac_ab_ij = int(a == 0 and b == 0)

    deriv[1:-1, 1:-1] = (dirac_ab_ijp1 + dirac_ab_ij) * 0.5
    return deriv


def del_ab_Id_ij(*, N, M, a, b):
    """
    This function returns the derivative of Id with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} Id_ij
    # Id_ij = u_{ij}
    # del_{i+a, j+b} Id_ij = del_{i+a, j+b} u_{ij}
    #     = dirac_{i+a, j+b; ij}
    #     = dirac_{a, b; 0,0}
    """

    assert_del_ab(a, b)

    deriv = np.zeros((N + 1, M + 1))
    deriv[1:-1, 1:-1] = int(a == 0 and b == 0)

    return deriv


# TODO: Seems unused. Remove if so. Naming is odd!
def del_ab_Id_ab(*, N, M, a, b):
    """
    This function returns the derivative of Id_{i+a,j+b} with respect to u_{i+a,j+b}.

    # deriv[i,j] = del_{i+a, j+b} Id_{i+a,j+b}
    # Id_{i+a,j+b} = u_{i+a,j+b}
    # del_{i+a, j+b} Id_{i+a,j+b} = del_{i+a, j+b} u_{i+a,j+b}
    #     = dirac_{i+a, j+b; i+a,j+b}
    #     = 1 sss (1 <= i+a <= N-1, and 1 <= j+b <= M-1)
    """

    assert_del_ab(a, b)

    # deriv[i,j]
    #     = del_{i+a,j+b} Id_{i+a, j+b}
    #     = 1 sss (1 <= i+a <= N-1 e 1 <= j+b <= M-1)

    deriv = np.zeros((N + 1, M + 1))
    deriv[1:-1, 1:-1] = np.ones((N - 1, M - 1))

    if a == 1:
        deriv[N - 1, :] = 0  # deriv[N-1,j] = del_{N,j+b} Id_{N,j+b} === 0
    elif a == -1:
        deriv[1, :] = 0  # deriv[1,j] = del_{0,j+b} Id_{0,j+b} === 0
    elif b == 1:
        deriv[:, M - 1] = 0
    elif b == -1:
        deriv[:, 1] = 0

    return deriv


def _create_lazy_immutable_property(prop_name, compute_func):
    """
    Creates a read-only property object that computes its value lazily
    using compute_func and caches the result.

    Args:
        prop_name (str): The name of the property (used for caching).
        compute_func (callable): A function (e.g., a lambda) that takes 'self'
                                 and returns the computed value for the property.

    Returns:
        property: A configured property object.
    """
    cache_attr = f"_cache_{prop_name}"

    def getter(self):
        # Check instance cache first
        # Use object.__getattribute__ to avoid triggering descriptor protocol recursively
        # if cache_attr itself becomes a property later (unlikely here -- X is a property of itself, which would cause this getter to call this getter, to call this getter, etc). Standard getattr is fine too (?).
        try:
            return object.__getattribute__(self, cache_attr)
        except AttributeError:
            # Value not cached, compute it
            value = compute_func(self)
            # Cache the computed value. Use object.__setattr__ to bypass
            # any potential custom __setattr__ on the instance (important for
            # immutability if there is a custom verifying that raises on
            # attempted mutation).
            object.__setattr__(self, cache_attr, value)
            return value

    # Create a read-only property (fset=None, fdel=None)
    return property(
        fget=getter,
        fset=None,
        fdel=None,
        doc=f"Lazy evaluated immutable property: {prop_name}",
    )


def _add_lazy_properties(cls):
    """
    Class decorator to add properties defined in _COMPUTED_PROPERTIES.

    Iterates over cls._COMPUTED_PROPERTIES and attaches a lazy, immutable
    property to the class for each entry.
    """
    # Get the definition dictionary from the class attribute
    computed_properties_dict = getattr(cls, "_COMPUTED_PROPERTIES", {})

    for name, func in computed_properties_dict.items():
        # Add the property to the class. It's bug if there is a clash.
        assert name not in cls.__dict__
        setattr(cls, name, _create_lazy_immutable_property(name, func))

    return cls


# TODO: This should work on a full grid. Not on hh and kk directly. Right? It takes hh, kk. Essentially, it's already takng a grid, just in a loose sense.
@_add_lazy_properties
class StateVars:
    """
    Immutable container for grid functions with lazily computed derived quantities.

    Input arrays (cp, T, cl, cd, cs) and parameters are stored on creation.
    Derived quantities (averages, derivatives, coefficients) are computed
    on first access via read-only properties and then cached.
    """

    # --- Definition of Computed Properties ---
    # Keys: Property names
    # Values: Lambdas taking 'self' to compute the value. Lambdas can access
    #         self.cp, self.T (input properties), self.hh (parameter properties),
    #         self._mc_Dl (bound helper methods), and other computed properties
    #         like self.Mxcp.
    _COMPUTED_PROPERTIES = {
        "MxT": lambda self: Mx_reg(self.T),
        "MyT": lambda self: My_reg(self.T),
        "Mxcp": lambda self: Mx_reg(self.cp),
        "Mycp": lambda self: My_reg(self.cp),
        "DmxT": lambda self: Dx_reg(self.T, self.hh),
        "DmyT": lambda self: Dy_reg(self.T, self.kk),
        "Dmxcl": lambda self: Dx_reg(self.cl, self.hh),
        "Dmycl": lambda self: Dy_reg(self.cl, self.kk),
        "Dmxcd": lambda self: Dx_reg(self.cd, self.hh),
        "Dmycd": lambda self: Dy_reg(self.cd, self.kk),
        # Dl = Dl(cp)
        "Dl_Mxcp": lambda self: self._model.Dl(self.Mxcp),
        "Dl_Mycp": lambda self: self._model.Dl(self.Mycp),
        "dDl_Mxcp": lambda self: self._model.Dl(self.Mxcp, d=1),
        "dDl_Mycp": lambda self: self._model.Dl(self.Mycp, d=1),
        # Vi = Vi(T)
        "V1T": lambda self: self._model.V1(self.T),
        "V2T": lambda self: self._model.V2(self.T),
        "dV1T": lambda self: self._model.V1(self.T, d=1),
        "dV2T": lambda self: self._model.V2(self.T, d=1),
        # Dd = Dd(cp, T)
        "Dd_MxcpT": lambda self: self._model.Dd(self.Mxcp, self.MxT),
        "Dd_MycpT": lambda self: self._model.Dd(self.Mycp, self.MyT),
        "delcp_Dd_MxcpT": lambda self: self._model.Dd(self.Mxcp, self.MxT, d=(1, 0)),
        "delcp_Dd_MycpT": lambda self: self._model.Dd(self.Mycp, self.MyT, d=(1, 0)),
        "delT_Dd_MxcpT": lambda self: self._model.Dd(self.Mxcp, self.MxT, d=(0, 1)),
        "delT_Dd_MycpT": lambda self: self._model.Dd(self.Mycp, self.MyT, d=(0, 1)),
    }

    def __init__(self, cp, T, cl, cd, cs, *, model: DefaultModel01, hh, kk):
        """
        Initializes the immutable pack with base grid functions and parameters.
        """

        # Store input data arrays directly (as read-only properties)
        # Use object.__setattr__ to bypass potential immutability checks during initialization
        # This is needed because properties are defined before __init__ completes
        # and may interfere with setting the initial values.
        # This is equivalent to setting attributes directly, but avoids potential conflicts
        # with property setters if immutability is enforced later.
        # Note: The properties themselves are defined by the @_add_lazy_properties decorator.
        #       Here we just set the underlying data attributes.

        object.__setattr__(self, "_cp_data", np.asarray(cp))
        object.__setattr__(self, "_T_data", np.asarray(T))
        object.__setattr__(self, "_cl_data", np.asarray(cl))
        object.__setattr__(self, "_cd_data", np.asarray(cd))
        object.__setattr__(self, "_cs_data", np.asarray(cs))

        # Store parameters
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_hh", hh)
        object.__setattr__(self, "_kk", kk)

        # Mark initialization as complete (for potential strict immutability via __setattr__)
        object.__setattr__(self, "_initialized", True)

    def into_dict(self, recipient: dict, which: List[str] = None):
        """
        Inserts the cp, T, cl, cd, cs variables in the `recipient` dictionary, together with the associated `_COMPUTED_PROPERTIES`. If you want a subset of all the properties, use the `which` parameter.

        Args:
            recipient (dict): The dictionary where the variables will be inserted.
            which (List[str], optional): A list of property names to include. If None, all properties are included. Defaults to None.

        Returns:
            dict: The updated recipient dictionary.

        """
        if which is None:
            which = self._COMPUTED_PROPERTIES
            which.extend(["cp", "T", "cl", "cd", "cs"])

        for prop_name in which:
            recipient[prop_name] = getattr(self, prop_name)

        return recipient

    # --- Read-only properties for direct access to input data/params ---
    # These are defined explicitly for clarity and direct access.
    @property
    def cp(self):
        return self._cp_data

    @property
    def T(self):
        return self._T_data

    @property
    def cl(self):
        return self._cl_data

    @property
    def cd(self):
        return self._cd_data

    @property
    def cs(self):
        return self._cs_data

    @property
    def model(self):
        return self._model

    @property
    def hh(self):
        return self._hh

    @property
    def kk(self):
        return self._kk

    # --- Enforce Immutability Strictly ---
    def __setattr__(self, name, value):
        # Allow setting cache attributes and during initialization
        if name.startswith("_cache_") or not getattr(self, "_initialized", False):
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"Cannot set attribute '{name}'. '{self.__class__.__name__}' instance is immutable."
            )

    def __delattr__(self, name):
        # Prevent deletion of attributes after initialization
        if name.startswith("_cache_") or not getattr(self, "_initialized", False):
            super().__delattr__(name)
        else:
            raise AttributeError(
                f"Cannot delete attribute '{name}'. '{self.__class__.__name__}' instance is immutable."
            )

    def with_changes(self, **kwargs):
        allowed_changes = ["cp", "T", "cl", "cd", "cs"]

        current = {term: getattr(self, term) for term in allowed_changes}

        for key in kwargs.keys():
            if key not in allowed_changes:
                raise ValueError(
                    f"{key}: Invalid change. Can only change: {allowed_changes}."
                )
            current[key] = kwargs[key]

        return StateVars(
            current["cp"],
            current["T"],
            current["cl"],
            current["cd"],
            current["cs"],
            model=self.model,
            hh=self.hh,
            kk=self.kk,
        )

    def copy(self):
        return self.with_changes()


def basic_inner_newton_step(*, x0, Jac, y):
    """
    x0, x1 em W_H
    y em inner(W_H)
    Jac em Lin(inner(W_H))

    x1(border) = x0(border)
    Jac (x1(inner) - x0(inner)) = y
    return x1
    """
    Np1, Mp1 = x0.shape
    N, M = Np1 - 1, Mp1 - 1
    assert y.shape == (N - 1, M - 1)
    assert Jac.shape == ((N - 1) * (M - 1), (N - 1) * (M - 1))
    x1 = np.copy(x0)
    x1[1:-1, 1:-1] += spla.spsolve(Jac, y.reshape((N - 1) * (M - 1))).reshape(
        (N - 1, M - 1)
    )
    return x1


def newton_step_inner_Fx_eq_C(*, x0, Fx0, JacFx0, C):
    """
    Newton step for solving: F(x) = C.

    Encontramos x1 tal que.

    Fx0 + JacFx0(x1 - x0) = C
    =>
        JacFx0 @ (x1 - x0) = C - Fx0
    =>
        x1 = x0 + JacFx0 \\ (C - Fx0)
    """

    x0 = np.asarray(x0).flatten("C")
    Fx0 = np.asarray(Fx0).flatten("C")
    C = np.asarray(C).flatten("C")

    dim = len(x0)
    assert len(C) == dim
    assert len(Fx0) == dim
    assert JacFx0.shape == (dim, dim)
    return x0 + spla.spsolve(JacFx0, C - Fx0)


class SemiDiscreteFieldBase(ABC):
    """
    A semi-discrete field, currently, is not exactly general. I'll follow the specific triangular form:

    Fcp = Fcp(*)
    FT = FT(cp, T, cs)
    Fcl = Fcl(cp, T, cl, cs)
    Fcd = Fcd(cp, T, cl, cd, cs)
    Fcs = Fcs(*)
    (where * means "potentially depend on all variables")

    Meaning we're assuming a type of triangular chaining in the coupling between T, Cl and Cd (for any fixed Cp,Cs pair). Abstract base methods for derivatives reflect this.
    """

    def __init__(self, *, grid: Grid, model: DefaultModel01):
        self._model = model
        self._grid = grid
        self._5dmatrix_factory = FiveDiagonalsMatrixFactory(grid.N, grid.M)

    @property
    def model(self):
        return self._model

    @property
    def grid(self):
        return self._grid

    @abstractmethod
    def Fcp(self, at_t: StateVars, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def FT(self, at_t: StateVars, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def Fcl(self, at_t: StateVars, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def Fcd(self, at_t: StateVars, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def Fcs(self, at_t: StateVars, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def delT_ab_FT_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of FcT with respect to T_{i+a,j+b}.

            deriv[i,j] = del_{T_{i+a,j+b}} FT_{i,j}
        """
        pass

    @abstractmethod
    def delT_ab_Fcl_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcl with respect to T_{i+a,j+b}.

            deriv[i,j] = del_{T_{i+a,j+b}} Fcl_{i,j}
        """
        pass

    @abstractmethod
    def delcl_ab_Fcl_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcl with respect to cl_{i+a,j+b}.

            deriv[i,j] = del_{cl_{i+a,j+b}} Fcl_{i,j}
        """
        pass

    @abstractmethod
    def delT_ab_Fcd_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcd with respect to T_{i+a,j+b}.

            deriv[i,j] = del_{T_{i+a,j+b}} Fcd_{i,j}
        """
        pass

    @abstractmethod
    def delcl_ab_Fcd_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcd with respect to cl_{i+a,j+b}.

            deriv[i,j] = del_{cl_{i+a,j+b}} Fcd_{i,j}
        """
        pass

    @abstractmethod
    def delcd_ab_Fcd_ij(self, gfp, *, a, b):
        """
        This function computes the derivative of Fcd with respect to cd_{i+a,j+b}.

            deriv[i,j] = del_{cd_{i+a,j+b}} Fcd_{i,j}
        """
        pass

    def delT_Fcl_at_w(self, state: StateVars, w):
        """
        This function computes the "Jacobian-vector" product of the "partial" Jacobian matrix del_T Fcl at `gfp` with the given vector w.
        """

        g = self.grid
        N = g.N
        M = g.M

        delT_Fcl = self._5dmatrix_factory.make_matrix(
            A_ij_ij=self.delT_ab_Fcl_ij(state, a=0, b=0)[1:-1, 1:-1],
            A_ij_ip1j=self.delT_ab_Fcl_ij(state, a=1, b=0)[1:-1, 1:-1],
            A_ij_im1j=self.delT_ab_Fcl_ij(state, a=-1, b=0)[1:-1, 1:-1],
            A_ij_ijp1=self.delT_ab_Fcl_ij(state, a=0, b=1)[1:-1, 1:-1],
            A_ij_ijm1=self.delT_ab_Fcl_ij(state, a=0, b=-1)[1:-1, 1:-1],
        )

        result = np.zeros_like(w)
        result[1:-1, 1:-1] = np.reshape(
            delT_Fcl @ np.reshape(w[1:-1, 1:-1], (N - 1) * (M - 1)), (N - 1, M - 1)
        )
        return result

    def delT_Fcd_at_w(self, state: StateVars, w):
        """
        This function computes the "Jacobian-vector" product of the "partial" Jacobian matrix del_T Fcd at `gfp` with the given vector w.
        """
        g = self.grid
        N = g.N
        M = g.M

        delT_Fcd = self._5dmatrix_factory.make_matrix(
            A_ij_ij=self.delT_ab_Fcd_ij(state, a=0, b=0)[1:-1, 1:-1],
            A_ij_ip1j=self.delT_ab_Fcd_ij(state, a=1, b=0)[1:-1, 1:-1],
            A_ij_im1j=self.delT_ab_Fcd_ij(state, a=-1, b=0)[1:-1, 1:-1],
            A_ij_ijp1=self.delT_ab_Fcd_ij(state, a=0, b=1)[1:-1, 1:-1],
            A_ij_ijm1=self.delT_ab_Fcd_ij(state, a=0, b=-1)[1:-1, 1:-1],
            rem_border_entries=False,
        )

        return np.reshape(delT_Fcd @ np.reshape(w, (N + 1) * (M + 1)), (N + 1, M + 1))

    def delcl_Fcd_at_w(self, state: StateVars, w):
        """
        This function computes the "Jacobian-vector" product of the "partial" Jacobian matrix del_cl Fcd at `gfp` with the given vector w.
        """
        g = self.grid
        N = g.N
        M = g.M

        delcl_Fcd = self._5dmatrix_factory.make_matrix(
            A_ij_ij=self.delcl_ab_Fcd_ij(state, a=0, b=0)[1:-1, 1:-1],
            A_ij_ip1j=self.delcl_ab_Fcd_ij(state, a=1, b=0)[1:-1, 1:-1],
            A_ij_im1j=self.delcl_ab_Fcd_ij(state, a=-1, b=0)[1:-1, 1:-1],
            A_ij_ijp1=self.delcl_ab_Fcd_ij(state, a=0, b=1)[1:-1, 1:-1],
            A_ij_ijm1=self.delcl_ab_Fcd_ij(state, a=0, b=-1)[1:-1, 1:-1],
            rem_border_entries=False,
        )

        return np.reshape(delcl_Fcd @ np.reshape(w, (N + 1) * (M + 1)), (N + 1, M + 1))


class ForcingTerms_CsTriple(ForcingTermsBase):
    def __init__(self, *, mms_case: MMSCaseBase, model: DefaultModel01):
        self._mms_case = mms_case
        self._model = model

    @property
    def grid(self):
        return self._mms_case.grid

    @property
    def mms_case(self):
        return self._mms_case

    @property
    def model(self):
        return self._model

    def fcp_ptwise(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        K1 = model.K1
        K2 = model.K2
        cp = mms_case.cp(t, xx, yy)
        dtCp = mms_case.dt_cp(t, xx, yy)
        cl = mms_case.cl(t, xx, yy)
        T = mms_case.T(t, xx, yy)
        return dtCp - (-cp * (K1 * (1 + cl) + K2 * T))

    # TODO: Some tests are failing after the introduction of this. Must fix. What could be? Failure has to do with expected orders not being attained. Re-check avg_int tests as well.
    def fcp(self, t, xx, yy):
        fcp_at_t = lambda xx, yy: self.fcp_ptwise(t, xx, yy)
        return avg_int(fcp_at_t, grid=Grid(xx, yy))

    # NOTE: If not careful, mail.py will be computing 2 avg integrals: one on top of the other. This needs careful attention.
    def fT(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        dtT = mms_case.dt_T(t, xx, yy)
        DT = model.DT
        K3 = model.K3
        lapT = mms_case.lap_T(t, xx, yy)
        T = mms_case.T(t, xx, yy)
        cp = mms_case.cp(t, xx, yy)
        return dtT - (DT * lapT - K3 * cp * T)

    def fcl(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        cp = mms_case.cp(t, xx, yy)
        dxCp = mms_case.dx_cp(t, xx, yy)
        dyCp = mms_case.dy_cp(t, xx, yy)

        T = mms_case.T(t, xx, yy)
        dxT = mms_case.dx_T(t, xx, yy)
        dyT = mms_case.dy_T(t, xx, yy)

        cl = mms_case.cl(t, xx, yy)
        dtCl = mms_case.dt_cl(t, xx, yy)
        lapCl = mms_case.lap_cl(t, xx, yy)
        dxCl = mms_case.dx_cl(t, xx, yy)
        dyCl = mms_case.dy_cl(t, xx, yy)

        K4 = model.K4

        V1 = model.V1(T)
        V2 = model.V2(T)
        dV1 = model.V1(T, d=1)
        dV2 = model.V2(T, d=1)

        Dl = model.Dl(cp)
        dDl = model.Dl(cp, d=1)

        return dtCl - (
            dDl * (dxCp * dxCl + dyCp * dyCl)
            + Dl * lapCl
            - V1 * dxCl
            - V2 * dyCl
            - (cl + 1) * (dV1 * dxT + dV2 * dyT)
            - K4 * cp * (cl + 1)
        )

    def fcd(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        cp = mms_case.cp(t, xx, yy)
        dxCp = mms_case.dx_cp(t, xx, yy)
        dyCp = mms_case.dy_cp(t, xx, yy)

        T = mms_case.T(t, xx, yy)
        dxT = mms_case.dx_T(t, xx, yy)
        dyT = mms_case.dy_T(t, xx, yy)

        cl = mms_case.cl(t, xx, yy)
        cs = mms_case.cs(t, xx, yy)

        cd = mms_case.cd(t, xx, yy)
        lapCd = mms_case.lap_cd(t, xx, yy)
        dxCd = mms_case.dx_cd(t, xx, yy)
        dyCd = mms_case.dy_cd(t, xx, yy)
        dtCd = mms_case.dt_cd(t, xx, yy)

        Kd = model.Kd
        Sd = model.Sd

        Dd = model.Dd(cp, T)
        dCp_Dd = model.Dd(cp, T, d=(1, 0))
        dT_Dd = model.Dd(cp, T, d=(0, 1))

        return dtCd - (
            (dCp_Dd * dxCp + dT_Dd * dxT) * dxCd
            + (dCp_Dd * dyCp + dT_Dd * dyT) * dyCd
            + Dd * lapCd
            + Kd * (Sd - cd) * (cl + 1) * cs
        )

    def fcs(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        Sd = model.Sd
        Kd = model.Kd
        dtCs = mms_case.dt_cs(t, xx, yy)
        cs = mms_case.cs(t, xx, yy)
        cl = mms_case.cl(t, xx, yy)
        cd = mms_case.cd(t, xx, yy)
        return dtCs - (-Kd * cs * (1 + cl) * (Sd - cd))


# This is similar, in spirit, to DefaultModel01. It's a standard form of field we're using in this work.
class SemiDiscreteField01_Base(SemiDiscreteFieldBase):
    """
    This is a particular kind of triangular field. It allows for some variability in how Cd/Cs interact in reactions.

    Fcp = -K1 cp (cl + 1) - K2 T cp + fcp
    FT = DT * Lap(T) - K3 T cp + fT
    Fcl = ∇·(Dl(cp) ∇cl) - ∇·(V(T)cl) - K4 cp (1 + cl) + fcl
    Fcd = ∇·(Dd(cp, T) ∇cd) + [Cs-Cd-int] + fcd
    Fcs = - [Cs-Cd-int] + fcs

    Where:
        - [Cs-Cd-int] represent a formula for the Cs/Cd interaction. Derived classes must specify [Cs-Cd-int]. [Cs-Cd-int] must be of the form:

        [Cs-Cd-int] = F1(Cp) * (a_T T + b_T) (a_Cl Cl + b_Cl) (a_Cd Cd + b_Cd) F2(Cs)

        where F1 and F2 could be any functions, and a,b,c,d,e,f could be any coefficients.

        - K1, K2, DT, K3, Dl, V, K4, Dd come from model terms.

        - fcp, fT, fcl, fcd, fcs are source forcing terms from MMS.

    A derived class, must implement:

        cscd_reaction_cp -- this is F1
        cscd_reaction_T -- returns the tuple (a_T,b_T), and this must be effectively constant (successive calls can't return different (a,b) values).
        cscd_reaction_cl -- returns (a_Cl,b_Cl); similar remarks as for T hold here.
        cscd_reaction_cd -- returns (a_Cd,b_Cd); similar remarks hold.
        cscd_reaction_cs -- this is F2.
    """

    def __init__(
        self, *, grid: Grid, model: DefaultModel01, forcing_terms: ForcingTermsBase
    ):
        super().__init__(grid=grid, model=model)

        # Import the forcing callables for convenience.
        forcing_terms_names = ["fcp", "fT", "fcl", "fcd", "fcs"]
        for term_name in forcing_terms_names:
            setattr(self, term_name, getattr(forcing_terms, term_name))

    @abstractmethod
    def cscd_reaction_cp(self, cp: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cscd_reaction_cs(self, cd: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cscd_reaction_T(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def cscd_reaction_cl(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def cscd_reaction_cd(self) -> Tuple[float, float]:
        pass

    def cscd_reaction_term(self, state: StateVars) -> np.ndarray:
        """
        This function computes the Cs/Cd interaction term.
        """
        cp_term = self.cscd_reaction_cp(state.cp)
        cs_term = self.cscd_reaction_cs(state.cs)
        a_T, b_T = self.cscd_reaction_T()
        a_cl, b_cl = self.cscd_reaction_cl()
        a_cd, b_cd = self.cscd_reaction_cd()
        reaction_term = (
            cp_term
            * (a_T * state.T + b_T)
            * (a_cl * state.cl + b_cl)
            * (a_cd * state.cd + b_cd)
            * cs_term
        )

        # This term must have null boundary conditions.
        # We currently force them. Should we assert instead? What bugs could this be masking? Check also the similar/associated functions.

        return reaction_term * self.grid.null_bd_mask

    def delT_ab_cscd_reaction_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of the Cs/Cd interaction term with respect to T_{i+a,j+b}.

            deriv[i,j] = del_{T_{i+a,j+b}} cscd_reaction_term_{i,j}
        """
        assert_del_ab(a, b)

        if a != 0 or b != 0:
            return self.grid.make_full0()

        cp_term = self.cscd_reaction_cp(state.cp)
        cs_term = self.cscd_reaction_cs(state.cs)
        a_T, _b_T_ignore = self.cscd_reaction_T()
        a_cl, b_cl = self.cscd_reaction_cl()
        a_cd, b_cd = self.cscd_reaction_cd()

        # For when the user explicitly sets this to 0.0, to skip the calculation.
        if a_T == 0.0:
            return self.grid.make_full0()

        return (
            cp_term
            * a_T
            * (a_cl * state.cl + b_cl)
            * (a_cd * state.cd + b_cd)
            * cs_term
            * self.grid.null_bd_mask
        )

    def delcl_ab_cscd_reaction_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of the Cs/Cd interaction term with respect to Cl_{i+a,j+b}.

            deriv[i,j] = del_{Cl_{i+a,j+b}} cscd_reaction_term_{i,j}
        """
        assert_del_ab(a, b)

        if a != 0 or b != 0:
            return self.grid.make_full0()

        cp_term = self.cscd_reaction_cp(state.cp)
        cs_term = self.cscd_reaction_cs(state.cs)
        a_T, b_T = self.cscd_reaction_T()
        a_cl, _b_cl_ignore = self.cscd_reaction_cl()
        a_cd, b_cd = self.cscd_reaction_cd()

        if a_cl == 0.0:
            return self.grid.make_full0()

        return (
            cp_term
            * (a_T * state.T + b_T)
            * a_cl
            * (a_cd * state.cd + b_cd)
            * cs_term
            * self.grid.null_bd_mask
        )

    def delcd_ab_cscd_reaction_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of the Cs/Cd interaction term with respect to Cd_{i+a,j+b}.

            deriv[i,j] = del_{Cd_{i+a,j+b}} cscd_reaction_term_{i,j}
        """
        assert_del_ab(a, b)

        if a != 0 or b != 0:
            return self.grid.make_full0()

        cp_term = self.cscd_reaction_cp(state.cp)
        cs_term = self.cscd_reaction_cs(state.cs)
        a_T, b_T = self.cscd_reaction_T()
        a_cl, b_cl = self.cscd_reaction_cl()
        a_cd, _b_cd_ignore = self.cscd_reaction_cd()

        if a_cd == 0.0:
            return self.grid.make_full0()

        return (
            cp_term
            * (a_T * state.T + b_T)
            * (a_cl * state.cl + b_cl)
            * a_cd
            * cs_term
            * self.grid.null_bd_mask
        )

    def Fcp(self, at_t: StateVars, t: float):
        """
        This function computes the value of Fcp.

        Fcp = -K1*(cl + 1)*cp - K2*T*cp
        """
        K1 = self.model.K1
        K2 = self.model.K2
        g = self.grid
        xx, yy = g.xx, g.yy

        the_Fcp = self.fcp(t, xx, yy)
        the_Fcp[1:-1, 1:-1] += (-K1 * (at_t.cl + 1) * at_t.cp - K2 * at_t.T * at_t.cp)[
            1:-1, 1:-1
        ]

        return the_Fcp

    def FT(self, at_t: StateVars, t: float):
        """
        This function computes the value of FT.

        FT = DT lap T - K3 cp T
        """

        g = self.grid
        xx, yy = g.xx, g.yy
        DT = self.model.DT
        K3 = self.model.K3

        the_FT = self.fT(t, xx, yy)
        the_FT[1:-1, 1:-1] += (
            g.Dx_star(DT * at_t.DmxT)
            + g.Dy_star(DT * at_t.DmyT)
            - K3 * at_t.cp * at_t.T
        )[1:-1, 1:-1]

        return the_FT

    def Fcl(self, at_t: StateVars, t: float):
        g = self.grid
        xx, yy = g.xx, g.yy
        K4 = self.model.K4

        the_Fcl = self.fcl(t, xx, yy)

        the_Fcl[1:-1, 1:-1] += (
            g.Dx_star(at_t.Dl_Mxcp * at_t.Dmxcl - Mx_reg(at_t.V1T * (at_t.cl + 1)))
            + g.Dy_star(at_t.Dl_Mycp * at_t.Dmycl - My_reg(at_t.V2T * (at_t.cl + 1)))
            - K4 * at_t.cp * (at_t.cl + 1)
        )[1:-1, 1:-1]

        return the_Fcl

    def Fcd(self, at_t, t):
        g = self.grid
        xx, yy = g.xx, g.yy

        the_Fcd = self.fcd(t, xx, yy)

        the_Fcd[1:-1, 1:-1] += (
            g.Dx_star(at_t.Dd_MxcpT * at_t.Dmxcd)
            + g.Dy_star(at_t.Dd_MycpT * at_t.Dmycd)
            + self.cscd_reaction_term(at_t)
        )[1:-1, 1:-1]

        return the_Fcd

    def Fcs(self, at_t, t):
        g = self.grid
        xx, yy = g.xx, g.yy
        return (
            self.fcs(t, xx, yy) - self.cscd_reaction_term(at_t)
        ) * self.grid.null_bd_mask

    def delT_ab_FT_ij(self, state: StateVars, *, a, b):
        g = self.grid
        DT = self.model.DT
        K3 = self.model.K3

        return DT * (
            g.del_ij_Dx_star_ij * g.del_ab_Dx_reg_ij(a, b)
            + g.del_ip1j_Dx_star_ij * g.del_ab_Dx_reg_ip1j(a, b)
            + g.del_ij_Dy_star_ij * g.del_ab_Dy_reg_ij(a, b)
            + g.del_ijp1_Dy_star_ij * g.del_ab_Dy_reg_ijp1(a, b)
        ) - K3 * state.cp * g.del_ab_Id_ij(a, b)

    def delT_ab_Fcl_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcl with respect to T_{i+a,j+b}.

            deriv[i,j] = del_{T_{i+a,j+b}} Fcl_{i,j}
        """

        g = self.grid
        cl_ab = grid0_fn_ab(state.cl, a, b)
        dV1T_ab = grid0_fn_ab(state.dV1T, a, b)
        dV2T_ab = grid0_fn_ab(state.dV2T, a, b)

        # Refactor this formula into separate x,y diffu and advec terms
        return (
            -(1 + cl_ab)
            * g.del_ab_Id_ab(a, b)
            * (
                dV1T_ab
                * (
                    g.del_ij_Dx_star_ij * g.del_ab_Mx_reg_ij(a, b)
                    + g.del_ip1j_Dx_star_ij * g.del_ab_Mx_reg_ip1j(a, b)
                )
                + dV2T_ab
                * (
                    g.del_ij_Dy_star_ij * g.del_ab_My_reg_ij(a, b)
                    + g.del_ijp1_Dy_star_ij * g.del_ab_My_reg_ijp1(a, b)
                )
            )
        )

    def delcl_ab_Fcl_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcl with respect to cl_{i+a,j+b}.

            deriv[i,j] = del_{cl_{i+a,j+b}} Fcl_{i,j}
        """
        g = self.grid
        K4 = self.model.K4

        Dl_Mxcp_ij = state.Dl_Mxcp
        Dl_Mxcp_ip1j = np.zeros_like(Dl_Mxcp_ij)
        Dl_Mxcp_ip1j[:-1, :] = Dl_Mxcp_ij[1:, :]

        Dl_Mycp_ij = state.Dl_Mycp
        Dl_Mycp_ijp1 = np.zeros_like(Dl_Mycp_ij)
        Dl_Mycp_ijp1[:, :-1] = Dl_Mycp_ij[:, 1:]

        V1T_ab = grid0_fn_ab(state.V1T, a, b)
        V2T_ab = grid0_fn_ab(state.V2T, a, b)

        # Refactor this formula into separate x,y diffu and advec terms, and reaction term.
        return (
            g.del_ij_Dx_star_ij * Dl_Mxcp_ij * g.del_ab_Dx_reg_ij(a, b)
            + g.del_ip1j_Dx_star_ij * Dl_Mxcp_ip1j * g.del_ab_Dx_reg_ip1j(a, b)
            + g.del_ij_Dy_star_ij * Dl_Mycp_ij * g.del_ab_Dy_reg_ij(a, b)
            + g.del_ijp1_Dy_star_ij * Dl_Mycp_ijp1 * g.del_ab_Dy_reg_ijp1(a, b)
            - g.del_ab_Id_ab(a, b)
            * (
                g.del_ij_Dx_star_ij * g.del_ab_Mx_reg_ij(a, b) * V1T_ab
                + g.del_ip1j_Dx_star_ij * g.del_ab_Mx_reg_ip1j(a, b) * V1T_ab
                + g.del_ij_Dy_star_ij * g.del_ab_My_reg_ij(a, b) * V2T_ab
                + g.del_ijp1_Dy_star_ij * g.del_ab_My_reg_ijp1(a, b) * V2T_ab
            )
            - K4 * state.cp * g.del_ab_Id_ij(a, b)
        )

    def delT_ab_Fcd_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcd with respect to T_{i+a,j+b}.

            deriv[i,j] = del_{T_{i+a,j+b}} Fcd_{i,j}
        """

        g = self.grid
        N = g.N
        M = g.M

        delT_Dd_MxcpT_ij = state.delT_Dd_MxcpT
        delT_Dd_MxcpT_ip1j = np.zeros((N + 1, M + 1))
        delT_Dd_MxcpT_ip1j[:-1, :] = delT_Dd_MxcpT_ij[1:, :]

        delT_Dd_MycpT_ij = state.delT_Dd_MycpT
        delT_Dd_MycpT_ijp1 = np.zeros((N + 1, M + 1))
        delT_Dd_MycpT_ijp1[:, :-1] = delT_Dd_MycpT_ij[:, 1:]

        Dmxcd_ij = state.Dmxcd
        Dmxcd_ip1j = np.zeros((N + 1, M + 1))
        Dmxcd_ip1j[:-1, :] = Dmxcd_ij[1:, :]

        Dmycd_ij = state.Dmycd
        Dmycd_ijp1 = np.zeros((N + 1, M + 1))
        Dmycd_ijp1[:, :-1] = Dmycd_ij[:, 1:]

        deriv_diffu_x = (
            g.del_ij_Dx_star_ij * Dmxcd_ij * delT_Dd_MxcpT_ij * g.del_ab_Mx_reg_ij(a, b)
            + g.del_ip1j_Dx_star_ij
            * Dmxcd_ip1j
            * delT_Dd_MxcpT_ip1j
            * g.del_ab_Mx_reg_ip1j(a, b)
        )

        deriv_diffu_y = (
            +g.del_ij_Dy_star_ij
            * Dmycd_ij
            * delT_Dd_MycpT_ij
            * g.del_ab_My_reg_ij(a, b)
            + g.del_ijp1_Dy_star_ij
            * Dmycd_ijp1
            * delT_Dd_MycpT_ijp1
            * g.del_ab_My_reg_ijp1(a, b)
        )

        deriv_cscd_reaction = self.delT_ab_cscd_reaction_ij(state, a=a, b=b)

        return deriv_diffu_x + deriv_diffu_y + deriv_cscd_reaction

    def delcl_ab_Fcd_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcd with respect to cl_{i+a,j+b}.

            deriv[i,j] = del_{cl_{i+a,j+b}} Fcd_{i,j}
        """
        # Fcd's difusion doesn't depend on cl. Only its reaction.
        return self.delcl_ab_cscd_reaction_ij(state, a=a, b=b)

    def delcd_ab_Fcd_ij(self, state: StateVars, *, a, b):
        """
        This function computes the derivative of Fcd with respect to cd_{i+a,j+b}.

            deriv[i,j] = del_{cd_{i+a,j+b}} Fcd_{i,j}
        """
        g = self.grid
        N = g.N
        M = g.M

        # Dd doesn't depend on cd. This facilitates calculations.

        Dd_MxcpT_ij = state.Dd_MxcpT
        Dd_MxcpT_ip1j = np.zeros((N + 1, M + 1))
        Dd_MxcpT_ip1j[:-1, :] = Dd_MxcpT_ij[1:, :]

        Dd_MycpT_ij = state.Dd_MycpT
        Dd_MycpT_ijp1 = np.zeros((N + 1, M + 1))
        Dd_MycpT_ijp1[:, :-1] = Dd_MycpT_ij[:, 1:]

        deriv_diffu_x = g.del_ij_Dx_star_ij * Dd_MxcpT_ij * g.del_ab_Dx_reg_ij(
            a, b
        ) + g.del_ip1j_Dx_star_ij * Dd_MxcpT_ip1j * g.del_ab_Dx_reg_ip1j(a, b)
        deriv_diffu_y = g.del_ij_Dy_star_ij * Dd_MycpT_ij * g.del_ab_Dy_reg_ij(
            a, b
        ) + g.del_ijp1_Dy_star_ij * Dd_MycpT_ijp1 * g.del_ab_Dy_reg_ijp1(a, b)
        deriv_cscd_reaction = self.delcd_ab_cscd_reaction_ij(state, a=a, b=b)

        return deriv_diffu_x + deriv_diffu_y + deriv_cscd_reaction


class SemiDiscreteField_CsTriple(SemiDiscreteField01_Base):
    """
    [Cs-Cd-int] = Kd (Sd - Cd) (1 + Cl) Cs

    Considering:
        [Cs-Cd-int] = F1(Cp) (a T + b) (c Cl + d) (e Cd + f) F2(Cs)
    we then have
        F1(Cp) = 1
        a = 0, b = 1
        d = c = 1
        e = -1
        f = Sd
        F2(Cs) = Kd Cs
    """

    # TODO: Take a ForcingTerms_CsTriple instead.
    def __init__(
        self, *, grid: Grid, model: DefaultModel01, forcing_terms: ForcingTermsBase
    ):
        super().__init__(grid=grid, model=model, forcing_terms=forcing_terms)

    def cscd_reaction_T(self):
        return (0, 1)

    def cscd_reaction_cl(self):
        return (1, 1)

    def cscd_reaction_cd(self):
        return (-1, self.model.Sd)

    def cscd_reaction_cp(self, cp):
        return self.grid.const_with_nullbd(1)

    def cscd_reaction_cs(self, cs):
        return self.model.Kd * cs


class TimeIntegratorBase(ABC):
    @abstractmethod
    def step(self, at_t0: StateVars, *, t0, dt):
        pass


class ForwardEulerIntegrator(TimeIntegratorBase):
    def __init__(self, semi_discrete_field: SemiDiscreteFieldBase, **kwargs):
        self.semi_discrete_field = semi_discrete_field

    def step(self, at_t0: StateVars, *, t0, dt):
        field = self.semi_discrete_field
        Fcp0 = field.Fcp(at_t0, t0)
        FT0 = field.FT(at_t0, t0)
        Fcl0 = field.Fcl(at_t0, t0)
        Fcd0 = field.Fcd(at_t0, t0)
        Fcs0 = field.Fcs(at_t0, t0)

        cp1 = at_t0.cp + dt * Fcp0
        T1 = at_t0.T + dt * FT0
        cl1 = at_t0.cl + dt * Fcl0
        cd1 = at_t0.cd + dt * Fcd0
        cs1 = at_t0.cs + dt * Fcs0

        return at_t0.with_changes(cp=cp1, T=T1, cl=cl1, cd=cd1, cs=cs1)


class P_ModifiedEuler_C_Trapezoidal_TimeIntegratorBase(ABC):
    """
    This is the main time integrator in this work. It emplyes a predictor-corrector (PC) scheme in which:

    1- cp,cs are predicted (abstract methods do this, for cs).
    2- Newton's method is used to predict T, cl, cd with the trapezoidal method.
    3- cp,cs are corrected (abstract methods do this, for cs).
    4- Go to (2-).

    The use of Newton's method to predict (T, cl, cd) relies on the "triangular" nature of the semi-discrete field (FT = FT(T), Fcl = Fcl(T, cl), Fcd = Fcd(T, cl, cd) -- assuming cp,cs are fixed).

    For now, in this work, I'm only interested in varying how Cs is handled in the field. This will be, generally speaking simpler. Most of the semi-discrete field definition is fixed. Therefore, much of the implicit calculations for time-integration are also fixed. Derived classes here mostly have to re-handle del_{cl} cd, del_{cd} cd, corrector for Cs.

    Derived classes must implement:

        initial_cp_step
        initial_cs_step
        corrector_cp_step
        corrector_cs_step
    """

    def __init__(
        self,
        semi_discrete_field: SemiDiscreteField_CsTriple,
        *,
        num_pc_steps=1,
        num_newton_steps=1,
    ):
        self.semi_discrete_field = semi_discrete_field
        self._model = semi_discrete_field.model
        grid = semi_discrete_field.grid
        self._grid = grid
        self._5dmatrix_factory = FiveDiagonalsMatrixFactory(grid.N, grid.M)
        self.num_pc_steps = num_pc_steps
        self.num_newton_steps = num_newton_steps

        # We'll keep last computed newton residuals in here. Key are integral variables: "T", "cl", "cd".
        self.last_residual: Dict = {}

    @abstractmethod
    def initial_cs_pred(self, at_t, t, *, dt):
        pass

    @abstractmethod
    def corrector_cs_step(self, T1, cl1, cd1, *, at_t0, t0, dt):
        pass

    def initial_cp_pred(self, at_t, t, *, dt):
        # Modified Euler, Heun
        Fcp = self.semi_discrete_field.Fcp

        Fcp0 = Fcp(at_t, t)
        cp_star = at_t.cp + dt * Fcp(at_t, t)
        gfp_star = at_t.with_changes(
            cp=cp_star, T=at_t.T, cl=at_t.cl, cd=at_t.cd, cs=at_t.cs
        )
        Fcp_star = Fcp(gfp_star, t + dt)

        cp1 = at_t.cp + 0.5 * dt * (Fcp0 + Fcp_star)
        return cp1

    def corrector_cp_step(self, T1, cl1, _cd1_ignroed, *, at_t0, t0, dt):
        assert dt > 0
        t1 = t0 + dt

        fcp = self.semi_discrete_field.fcp
        g = self.semi_discrete_field.grid
        K1 = self._model.K1
        K2 = self._model.K2

        cp0_int = at_t0.cp[1:-1, 1:-1]
        cl0_int = at_t0.cl[1:-1, 1:-1]
        T0_int = at_t0.T[1:-1, 1:-1]
        T1_int = T1[1:-1, 1:-1]
        cl1_int = cl1[1:-1, 1:-1]

        src0_int = fcp(t0, g.xx, g.yy)[1:-1, 1:-1]
        src1_int = fcp(t1, g.xx, g.yy)[1:-1, 1:-1]

        alpha0 = -K2 * T0_int - K1 * (cl0_int + 1)
        alpha1 = -K2 * T1_int - K1 * (cl1_int + 1)

        trap_rhs_num = (1 + (dt / 2) * alpha0) * cp0_int + (dt / 2) * (
            src0_int + src1_int
        )
        trap_rhs_denom = 1 - (dt / 2) * alpha1

        cp1 = np.zeros_like(at_t0.cp)
        cp1[1:-1, 1:-1] = trap_rhs_num / trap_rhs_denom

        return cp1

    def newton_step_T(self, at_t0: StateVars, *, t0, dt, YT0):
        """
        Faz uma iteração do método de Newton para a temperatura.
        """
        assert dt > 0
        t1 = t0 + dt

        field = self.semi_discrete_field
        grid = field.grid
        make_5d_matrix = self._5dmatrix_factory.make_matrix
        FT = field.FT

        # GT(T) = 2 T - FT(cp1_pred, T, t1).
        JacGT_ij_ij = 2 - dt * field.delT_ab_FT_ij(at_t0, a=0, b=0)
        JacGT_ij_ip1j = -dt * field.delT_ab_FT_ij(at_t0, a=1, b=0)
        JacGT_ij_im1j = -dt * field.delT_ab_FT_ij(at_t0, a=-1, b=0)
        JacGT_ij_ijp1 = -dt * field.delT_ab_FT_ij(at_t0, a=0, b=1)
        JacGT_ij_ijm1 = -dt * field.delT_ab_FT_ij(at_t0, a=0, b=-1)

        delT_GT = make_5d_matrix(
            A_ij_ij=JacGT_ij_ij[1:-1, 1:-1],
            A_ij_ip1j=JacGT_ij_ip1j[1:-1, 1:-1],
            A_ij_im1j=JacGT_ij_im1j[1:-1, 1:-1],
            A_ij_ijp1=JacGT_ij_ijp1[1:-1, 1:-1],
            A_ij_ijm1=JacGT_ij_ijm1[1:-1, 1:-1],
        )

        GT1_0 = 2 * at_t0.T - dt * FT(at_t0, t1)

        inner_T0_flat = at_t0.T[1:-1, 1:-1].flatten("C")
        inner_GT1_0_flat = GT1_0[1:-1, 1:-1].flatten("C")
        inner_YT0_flat = YT0[1:-1, 1:-1].flatten("C")

        T_next_inner_flat = newton_step_inner_Fx_eq_C(
            x0=inner_T0_flat,
            JacFx0=delT_GT,
            C=inner_YT0_flat,
            Fx0=inner_GT1_0_flat,
        )

        T_next = grid.make_full0()
        T_next[1:-1, 1:-1] = np.reshape(T_next_inner_flat, grid.interior_shape)

        at_t1 = at_t0.with_changes(T=T_next)
        GT1_1 = 2 * T_next - dt * FT(at_t1, t1)
        self.last_residual["T"] = GT1_1 - YT0

        return T_next

    def newton_step_cl(self, at_t0, T1, *, t0, dt, Ycl0):
        assert dt > 0
        t1 = t0 + dt

        field = self.semi_discrete_field
        Fcl = field.Fcl

        delcl_Gcl_ij_ij = 2 - dt * field.delcl_ab_Fcl_ij(at_t0, a=0, b=0)
        delcl_Gcl_ij_ip1j = -dt * field.delcl_ab_Fcl_ij(at_t0, a=1, b=0)
        delcl_Gcl_ij_ijm1 = -dt * field.delcl_ab_Fcl_ij(at_t0, a=0, b=-1)
        delcl_Gcl_ij_im1j = -dt * field.delcl_ab_Fcl_ij(at_t0, a=-1, b=0)
        delcl_Gcl_ij_ijp1 = -dt * field.delcl_ab_Fcl_ij(at_t0, a=0, b=1)

        delcl_Gcl = self._5dmatrix_factory.make_matrix(
            A_ij_ij=delcl_Gcl_ij_ij[1:-1, 1:-1],
            A_ij_ip1j=delcl_Gcl_ij_ip1j[1:-1, 1:-1],
            A_ij_im1j=delcl_Gcl_ij_im1j[1:-1, 1:-1],
            A_ij_ijp1=delcl_Gcl_ij_ijp1[1:-1, 1:-1],
            A_ij_ijm1=delcl_Gcl_ij_ijm1[1:-1, 1:-1],
        )

        rhs = (
            Ycl0
            - 2 * at_t0.cl
            + dt * Fcl(at_t0, t1)
            + dt * field.delT_Fcl_at_w(at_t0, T1 - at_t0.T)
        )[1:-1, 1:-1]

        cl1 = basic_inner_newton_step(x0=at_t0.cl, Jac=delcl_Gcl, y=rhs)
        at_t1 = at_t0.with_changes(T=T1, cl=cl1)
        Gcl1_1 = 2 * cl1 - dt * Fcl(at_t1, t1)
        self.last_residual["cl"] = Gcl1_1 - Ycl0

        return cl1

    def newton_step_cd(self, at_t0, T1, cl1, *, t0, dt, Ycd0):
        assert dt > 0
        t1 = t0 + dt

        field = self.semi_discrete_field

        delcd_Gcd_ij_ij = 2 - dt * field.delcd_ab_Fcd_ij(at_t0, a=0, b=0)
        delcd_Gcd_ij_ip1j = -dt * field.delcd_ab_Fcd_ij(at_t0, a=1, b=0)
        delcd_Gcd_ij_ijm1 = -dt * field.delcd_ab_Fcd_ij(at_t0, a=0, b=-1)
        delcd_Gcd_ij_im1j = -dt * field.delcd_ab_Fcd_ij(at_t0, a=-1, b=0)
        delcd_Gcd_ij_ijp1 = -dt * field.delcd_ab_Fcd_ij(at_t0, a=0, b=1)

        delcd_Gcd = self._5dmatrix_factory.make_matrix(
            A_ij_ij=delcd_Gcd_ij_ij[1:-1, 1:-1],
            A_ij_ip1j=delcd_Gcd_ij_ip1j[1:-1, 1:-1],
            A_ij_im1j=delcd_Gcd_ij_ijm1[1:-1, 1:-1],
            A_ij_ijp1=delcd_Gcd_ij_ijp1[1:-1, 1:-1],
            A_ij_ijm1=delcd_Gcd_ij_im1j[1:-1, 1:-1],
        )

        rhs = (
            Ycd0
            - 2 * at_t0.cd
            + dt * field.Fcd(at_t0, t1)
            + dt * field.delT_Fcd_at_w(at_t0, T1 - at_t0.T)
            + dt * field.delcl_Fcd_at_w(at_t0, cl1 - at_t0.cl)
        )[1:-1, 1:-1]

        cd1 = basic_inner_newton_step(x0=at_t0.cd, Jac=delcd_Gcd, y=rhs)
        at_t1 = at_t0.with_changes(T=T1, cl=cl1, cd=cd1)
        Gcd1_1 = 2 * cd1 - dt * field.Fcd(at_t1, t1)
        self.last_residual["cd"] = Gcd1_1 - Ycd0

        return cd1

    def step(self, at_t0: StateVars, *, t0, dt):
        assert dt > 0

        field = self.semi_discrete_field

        YT0 = dt * field.FT(at_t0, t0) + 2 * at_t0.T
        Ycl0 = dt * field.Fcl(at_t0, t0) + 2 * at_t0.cl
        Ycd0 = dt * field.Fcd(at_t0, t0) + 2 * at_t0.cd

        cp1 = self.initial_cp_pred(at_t0, t0, dt=dt)
        T1 = at_t0.T
        cl1 = at_t0.cl
        cd1 = at_t0.cd
        cs1 = self.initial_cs_pred(at_t0, t0, dt=dt)

        for _pc_j in range(self.num_pc_steps):
            # Cp, Cs is fixed at the previous prediction/correction.
            # We iterate on T,Cl,Cd with such Cp, Cs fixed.
            for _newt_j in range(self.num_newton_steps):
                CpCsPred_TClCd0 = at_t0.with_changes(
                    cp=cp1, T=T1, cl=cl1, cd=cd1, cs=cs1
                )
                T1 = self.newton_step_T(CpCsPred_TClCd0, t0=t0, dt=dt, YT0=YT0)
                cl1 = self.newton_step_cl(CpCsPred_TClCd0, T1, t0=t0, dt=dt, Ycl0=Ycl0)
                cd1 = self.newton_step_cd(
                    CpCsPred_TClCd0, T1, cl1, t0=t0, dt=dt, Ycd0=Ycd0
                )

            # Correct Cp,Cs for the next newton iteration.
            cp1 = self.corrector_cp_step(T1, cl1, cd1, at_t0=at_t0, t0=t0, dt=dt)
            cs1 = self.corrector_cs_step(T1, cl1, cd1, at_t0=at_t0, t0=t0, dt=dt)

        return at_t0.with_changes(cp=cp1, T=T1, cl=cl1, cd=cd1, cs=cs1)


class P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_CsTriple(
    P_ModifiedEuler_C_Trapezoidal_TimeIntegratorBase
):
    """
    Trapezoidal PC Stepper for the CsTriple semi-discrete field.
    Modified Euler is used for Cp,Cs initial predictions.
    Regular Trapezoidal is used for Cp,Cs corrections.
    """

    def __init__(
        self,
        semi_discrete_field: SemiDiscreteField_CsTriple,
        *,
        num_pc_steps=1,
        num_newton_steps=1,
    ):
        super().__init__(
            semi_discrete_field=semi_discrete_field,
            num_pc_steps=num_pc_steps,
            num_newton_steps=num_newton_steps,
        )

    def initial_cs_pred(self, at_t, t, *, dt):
        # Modified Euler, Heun

        Fcs = self.semi_discrete_field.Fcs

        Fcs0 = Fcs(at_t, t)

        cs_star = at_t.cs + dt * Fcs(at_t, t)
        gfp_star = at_t.with_changes(
            cp=at_t.cp, T=at_t.T, cl=at_t.cl, cd=at_t.cd, cs=cs_star
        )
        Fcs_star = Fcs(gfp_star, t + dt)

        cs1 = at_t.cs + 0.5 * dt * (Fcs0 + Fcs_star)
        return cs1

    def corrector_cs_step(self, _T1_ignored, cl1, cd1, *, at_t0, t0, dt):
        assert dt > 0
        t1 = t0 + dt

        fcs = self.semi_discrete_field.fcs
        g = self._grid

        Sd = self._model.Sd
        Kd = self._model.Kd

        cs0_int = at_t0.cs[1:-1, 1:-1]
        cd0_int = at_t0.cd[1:-1, 1:-1]
        cl0_int = at_t0.cl[1:-1, 1:-1]
        cd1_int = cd1[1:-1, 1:-1]
        cl1_int = cl1[1:-1, 1:-1]

        src0_int = fcs(t0, g.xx, g.yy)[1:-1, 1:-1]
        src1_int = fcs(t1, g.xx, g.yy)[1:-1, 1:-1]

        alpha0 = -Kd * (Sd - cd0_int) * (1 + cl0_int)
        alpha1 = -Kd * (Sd - cd1_int) * (1 + cl1_int)

        trap_rhs_num = (1 + (dt / 2) * alpha0) * cs0_int + (dt / 2) * (
            src0_int + src1_int
        )
        trap_rhs_denom = 1 - (dt / 2) * alpha1

        cs1 = np.zeros_like(at_t0.cs)
        cs1[1:-1, 1:-1] = trap_rhs_num / trap_rhs_denom
        return cs1


class ForcingTerms_HCsTriple(ForcingTermsBase):
    def __init__(self, *, mms_case: MMSCaseBase, model: DefaultModel01):
        self.cs3_forcing_terms = ForcingTerms_CsTriple(mms_case=mms_case, model=model)
        self._mms_case = mms_case
        self._model = model

        # NOTE: Ideally, forcing terms should be fully symbolic. In this case, I couldn't make this work. Grid information is then needed.
        self._grid = mms_case.grid

    @property
    def grid(self):
        return self._grid

    def fcp(self, t, xx, yy):
        return self.cs3_forcing_terms.fcp(t, xx, yy)

    def fT(self, t, xx, yy):
        return self.cs3_forcing_terms.fT(t, xx, yy)

    def fcl(self, t, xx, yy):
        return self.cs3_forcing_terms.fcl(t, xx, yy)

    @property
    def mms_case(self):
        return self._mms_case

    @property
    def model(self):
        return self._model

    def fcd(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        cp = mms_case.cp(t, xx, yy)
        dxCp = mms_case.dx_cp(t, xx, yy)
        dyCp = mms_case.dy_cp(t, xx, yy)

        T = mms_case.T(t, xx, yy)
        dxT = mms_case.dx_T(t, xx, yy)
        dyT = mms_case.dy_T(t, xx, yy)

        cl = mms_case.cl(t, xx, yy)
        cs = mms_case.cs(t, xx, yy)

        cd = mms_case.cd(t, xx, yy)
        lapCd = mms_case.lap_cd(t, xx, yy)
        dxCd = mms_case.dx_cd(t, xx, yy)
        dyCd = mms_case.dy_cd(t, xx, yy)
        dtCd = mms_case.dt_cd(t, xx, yy)

        Kd = model.Kd
        Sd = model.Sd

        Dd = model.Dd(cp, T)
        dCp_Dd = model.Dd(cp, T, d=(1, 0))
        dT_Dd = model.Dd(cp, T, d=(0, 1))

        return dtCd - (
            (dCp_Dd * dxCp + dT_Dd * dxT) * dxCd
            + (dCp_Dd * dyCp + dT_Dd * dyT) * dyCd
            + Dd * lapCd
            + Kd * (Sd - cd) * (cl + 1) * (cs > 0)
        )

    def fcs(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        Sd = model.Sd
        Kd = model.Kd
        dtCs = mms_case.dt_cs(t, xx, yy)
        cs = mms_case.cs(t, xx, yy)
        cl = mms_case.cl(t, xx, yy)
        cd = mms_case.cd(t, xx, yy)
        return dtCs - (-Kd * (cs > 0) * (1 + cl) * (Sd - cd))


class SemiDiscreteField_HCsTriple(SemiDiscreteField01_Base):
    """
    [Cs-Cd-int] = Kd (Sd - Cd) (1 + Cl) H(Cs)
    Considering:
        [Cs-Cd-int] = F1(Cp) (a T + b) (c Cl + d) (e Cd + f) F2(Cs)
    we then have
        F1(Cp) = 1
        a = 0, b = 1
        d = c = 1
        e = -1
        f = Sd
        F2(Cs) = Kd H(Cs)
    """

    # TODO: Take a ForcingTerms_CsTriple instead.
    def __init__(
        self, *, grid: Grid, model: DefaultModel01, forcing_terms: ForcingTermsBase
    ):
        super().__init__(grid=grid, model=model, forcing_terms=forcing_terms)

    def cscd_reaction_T(self):
        return (0, 1)

    def cscd_reaction_cl(self):
        return (1, 1)

    def cscd_reaction_cd(self):
        return (-1, self.model.Sd)

    def cscd_reaction_cp(self, cp):
        return self.grid.const_with_nullbd(1)

    def cscd_reaction_cs(self, cs):
        return self.model.Kd * (cs > 0)


class P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_HCsTriple(
    P_ModifiedEuler_C_Trapezoidal_TimeIntegratorBase
):
    """
    This is the main time integrator in this work. It emplyes a predictor-corrector (PC) scheme in which:

    1- cp,cs are predicted with the modified euler mehtod.
    2- Newton's method is used to predict T, cl, cd with the trapezoidal method.
    3- cp,cs are corrected with the trapezoidal method.
    4- Go to (2-).

    The use of Newton's method to predict (T, cl, cd) relies on the "triangular" nature of the semi-discrete field (FT = FT(T), Fcl = Fcl(T, cl), Fcd = Fcd(T, cl, cd) -- assuming cp,cs are fixed).

    The difference is that this is for the HCsTriple field.
    """

    def __init__(
        self,
        semi_discrete_field: SemiDiscreteField_HCsTriple,
        *,
        num_pc_steps=1,
        num_newton_steps=1,
    ):
        super().__init__(
            semi_discrete_field=semi_discrete_field,
            num_newton_steps=num_newton_steps,
            num_pc_steps=num_pc_steps,
        )

    def initial_cs_pred(self, at_t, t, *, dt):
        # Modified Euler, Heun, adapted to force concentrations back to 0 if negative.

        Fcs = self.semi_discrete_field.Fcs

        Fcs0 = Fcs(at_t, t)

        cs_star = at_t.cs + dt * Fcs(at_t, t)
        gfp_star = at_t.with_changes(
            cp=at_t.cp, T=at_t.T, cl=at_t.cl, cd=at_t.cd, cs=cs_star
        )
        Fcs_star = Fcs(gfp_star, t + dt)

        cs1 = at_t.cs + (0.5 * dt) * (Fcs0 + Fcs_star)
        return cs1 * self._grid.null_bd_mask

    def corrector_cs_step(self, _T1_ignored, cl1, cd1, *, at_t0, t0, dt):
        # Adapted trapezoidal for the forced concentrations. The method:
        #
        # We want to solve for Cs1 such that
        #
        # 2Cs1 - dt Kd (Sd - Cd1) (1 + Cl1) H(Cs1) = Y0
        #
        # For Y0 = 2Cs0 + dt Fcs(at_t0) + dt fcs1
        #   for previously computed Cs0, and forcing term fcs1.
        #
        # Method idea:
        #   - Say we found our Cs1 Solution.
        #   - Set R1 = Kd (Sd - Cd1) (1 + Cl1). We work under the regiment of (2 - dt R1) > 0. This is required for proving that what follows actually work.
        #   - Consider the 9 cases analysis sign(Y0) in (-1,0,1) and sign(Cs1) in (-1,0,1).
        #   - It is possible to prove sign(Y0) = sign(Cs1)
        #   - We can solve for Cs1, obtaining:
        #       - If Y0 > 0: Cs1 = Y0/(2 - dt R1)
        #       - If Y0 = 0: Cs1 = 0
        #       - If Y0 < 0: Cs1 = Y0/2

        essentially_zero_tol = np.finfo(float).eps * 100

        Fcs = self.semi_discrete_field.Fcs
        fcs = self.semi_discrete_field.fcs
        model = self._model
        grid = self._grid
        xx, yy = grid.xx, grid.yy

        Kd = model.Kd
        Sd = model.Sd
        fcs1 = fcs(t0 + dt, xx, yy)

        R1 = (Sd - cd1) * (1 + cl1) * Kd
        del_Y1 = 2 - dt * R1

        # The proof that this method work with the explained (in the comments above) case analysis relies on the positivity of del_Y1 throughout the whole interior domain.
        if np.any(del_Y1 < essentially_zero_tol):
            raise ValueError(
                "Denominator 2 - Δt Kd (Sd - Cd1) (1 + Cl1) below positiveness treshold."
            )

        Y0 = 2 * at_t0.cs + dt * Fcs(at_t0, t0) + dt * fcs1

        where_pos = Y0 > essentially_zero_tol
        where_neg = Y0 < -essentially_zero_tol
        cs1 = self._grid.make_full0()
        cs1[where_pos] = Y0[where_pos] / del_Y1[where_pos]
        cs1[where_neg] = Y0[where_neg] / 2.0

        return cs1 * self._grid.null_bd_mask


def state_from_mms_when(*, mms_case, t, grid):
    """    This function returns a StateVars object with the values of the MMS solution at time t.
    """

    model = mms_case.model
    xx, yy = grid.xx, grid.yy

    cp_t_exact = mms_case.cp(t, xx, yy)
    T_t_exact = mms_case.T(t, xx, yy)
    cl_t_exact = mms_case.cl(t, xx, yy)
    cd_t_exact = mms_case.cd(t, xx, yy)
    cs_t_exact = mms_case.cs(t, xx, yy)

    return StateVars(
        cp_t_exact, T_t_exact, cl_t_exact, cd_t_exact, cs_t_exact,
        model=model, hh=grid.hh, kk=grid.kk
    )


def heaviside_regularized(x: np.ndarray, regularization_factor: float) -> np.ndarray:
    """
    Regularized Heaviside function.

    H_η(x) = 1 / (1 + exp(-η x))

    Args:
        x: np.ndarray
        regularization_factor: float
            The regularization factor for the Heaviside term, η.

    Returns:
        np.ndarray
    """
    return 1 / (1 + np.exp(-regularization_factor * x))

class ForcingTerms_RegHCsTriple(ForcingTermsBase):
    """
    This class is used to compute the forcing terms for the HCsTriple field.
    """

    def __init__(self, *, mms_case: MMSCaseBase, model: DefaultModel01, regularization_factor: float):
        """
        The regularization factor for the Heaviside term is specified by the `regularization_factor` argument.
        """
        self.cs3_forcing_terms = ForcingTerms_CsTriple(mms_case=mms_case, model=model)
        self._mms_case = mms_case
        self._model = model
        self._regularization_factor = regularization_factor

    def fcp(self, t, xx, yy):
        return self.cs3_forcing_terms.fcp(t, xx, yy)

    def fT(self, t, xx, yy):
        return self.cs3_forcing_terms.fT(t, xx, yy)

    def fcl(self, t, xx, yy):
        return self.cs3_forcing_terms.fcl(t, xx, yy)

    @property
    def mms_case(self):
        return self._mms_case

    @property
    def model(self):
        return self._model

    @property
    def regularization_factor(self):
        return self._regularization_factor

    def fcd(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        cp = mms_case.cp(t, xx, yy)
        dxCp = mms_case.dx_cp(t, xx, yy)
        dyCp = mms_case.dy_cp(t, xx, yy)

        T = mms_case.T(t, xx, yy)
        dxT = mms_case.dx_T(t, xx, yy)
        dyT = mms_case.dy_T(t, xx, yy)

        cl = mms_case.cl(t, xx, yy)
        cs = mms_case.cs(t, xx, yy)

        cd = mms_case.cd(t, xx, yy)
        lapCd = mms_case.lap_cd(t, xx, yy)
        dxCd = mms_case.dx_cd(t, xx, yy)
        dyCd = mms_case.dy_cd(t, xx, yy)
        dtCd = mms_case.dt_cd(t, xx, yy)

        Kd = model.Kd
        Sd = model.Sd

        Dd = model.Dd(cp, T)
        dCp_Dd = model.Dd(cp, T, d=(1, 0))
        dT_Dd = model.Dd(cp, T, d=(0, 1))

        RegHCs = heaviside_regularized(cs, self.regularization_factor)

        return dtCd - (
            (dCp_Dd * dxCp + dT_Dd * dxT) * dxCd
            + (dCp_Dd * dyCp + dT_Dd * dyT) * dyCd
            + Dd * lapCd
            + Kd * (Sd - cd) * (cl + 1) * RegHCs
        )

    def fcs(self, t, xx, yy):
        mms_case = self.mms_case
        model = self.model

        Sd = model.Sd
        Kd = model.Kd
        dtCs = mms_case.dt_cs(t, xx, yy)
        cs = mms_case.cs(t, xx, yy)
        cl = mms_case.cl(t, xx, yy)
        cd = mms_case.cd(t, xx, yy)
        RegHCs = heaviside_regularized(cs, self.regularization_factor)
        return dtCs - (-Kd * (1 + cl) * (Sd - cd) * RegHCs)

class SemiDiscreteField_RegHCsTriple(SemiDiscreteField01_Base):
    """
    [Cs-Cd-int] = Kd (Sd - Cd) (1 + Cl) H_η(Cs)
    Considering:
        [Cs-Cd-int] = F1(Cp) (a T + b) (c Cl + d) (e Cd + f) F2(Cs)
    we then have
        F1(Cp) = 1
        a = 0, b = 1
        d = c = 1
        e = -1
        f = Sd
        F2(Cs) = Kd H_η(Cs)

    Remark:
        The regularization factor for the Heaviside term is specified by the `regularization_factor` argument in the constructor.
    """

    def __init__(
        self, *, grid: Grid, model: DefaultModel01, forcing_terms: ForcingTermsBase, regularization_factor: float
    ):
        super().__init__(grid=grid, model=model, forcing_terms=forcing_terms)
        self._regularization_factor = regularization_factor

    @property
    def regularization_factor(self):
        return self._regularization_factor

    def cscd_reaction_T(self):
        return (0, 1)

    def cscd_reaction_cl(self):
        return (1, 1)

    def cscd_reaction_cd(self):
        return (-1, self.model.Sd)

    def cscd_reaction_cp(self, cp):
        return self.grid.const_with_nullbd(1)

    def cscd_reaction_cs(self, cs):
        return self.model.Kd * heaviside_regularized(cs, self.regularization_factor)


class P_ModifiedEuler_C_Trapezoidal_TimeIntegrator_RegHCsTriple(
    P_ModifiedEuler_C_Trapezoidal_TimeIntegratorBase
):
    """
    This is the main time integrator in this work. It emplyes a predictor-corrector (PC) scheme in which:

    1- cp,cs are predicted with the modified euler mehtod.
    2- Newton's method is used to predict T, cl, cd with the trapezoidal method.
    3- cp,cs are corrected with the trapezoidal method.
    4- Go to (2-).

    The use of Newton's method to predict (T, cl, cd) relies on the "triangular" nature of the semi-discrete field (FT = FT(T), Fcl = Fcl(T, cl), Fcd = Fcd(T, cl, cd) -- assuming cp,cs are fixed).

    The difference is that this is for the RegHCsTriple field.
    """

    def __init__(
        self,
        semi_discrete_field: SemiDiscreteField_RegHCsTriple,
        *,
        num_pc_steps=1,
        num_newton_steps=1, # For triangular semidiscrete field.
        regularization_factor: float,
        num_newton_iterations: int = 5,  # For the implicit Cs piece.
        consec_xs_rtol: float = 1e-6 # Consecutive xs relative tolerance. Set to 0 to disable.
    ):
        super().__init__(
            semi_discrete_field=semi_discrete_field,
            num_newton_steps=num_newton_steps,
            num_pc_steps=num_pc_steps,
        )
        self._regularization_factor = regularization_factor
        self._num_newton_iterations = num_newton_iterations
        self._consec_xs_rtol = consec_xs_rtol

    def initial_cs_pred(self, at_t, t, *, dt):
        # Modified Euler, Heun, adapted to force concentrations back to 0 if negative.

        Fcs = self.semi_discrete_field.Fcs

        Fcs0 = Fcs(at_t, t)

        cs_star = at_t.cs + dt * Fcs(at_t, t)
        gfp_star = at_t.with_changes(
            cp=at_t.cp, T=at_t.T, cl=at_t.cl, cd=at_t.cd, cs=cs_star
        )
        Fcs_star = Fcs(gfp_star, t + dt)

        cs1 = at_t.cs + (0.5 * dt) * (Fcs0 + Fcs_star)
        return cs1 * self._grid.null_bd_mask

    def _predictor_equation(self, x: np.ndarray, y: np.ndarray, a: np.ndarray, eta: float):
        return 2*x + (2*x - y) * np.exp(-eta * x) - y + a

    def _predictor_jacobian(self, x: np.ndarray, y: np.ndarray, a: np.ndarray, eta: float):
        exp_term = np.exp(-eta * x)
        return 2 + 2 * exp_term - eta * (2*x - y) * exp_term

    def _newton_iterations(self, y: np.ndarray, a: np.ndarray, eta: float, x0: np.ndarray):
        x = x0
        for _ in range(self._num_newton_iterations):
            f = self._predictor_equation(x, y, a, eta)
            J = self._predictor_jacobian(x, y, a, eta)
            dx = -f / J ## How to handle the case when J is 0?
            x = x + dx
            if np.all(np.max(np.abs(dx)) < self._consec_xs_rtol * np.abs(x)):
                break
        return x

    def corrector_cs_step(self, _T1_ignored, cl1, cd1, *, at_t0, t0, dt):
        """
        This method solves the following implicit equation for cs1:

        2cs1 + dt * Kd * (Sd - cd1) * (cl1 + 1) * H_η(cs1)
            = 2cs0
                - dt * Kd * (Sd - cd0) * (cl0 + 1) * H_η(cs0)
                + dt * (fcs(t0) + fcs(t0 + dt))

        which can be rewritten in a more algebraically convenient form as:

            2x + (2x - y) e^(-η x) = y - a

        where

            x = cs1
            y = 2cs0
                - dt * Kd * (Sd - cd0) * (cl0 + 1) * H_η(cs0)
                + dt * (fcs(t0) + fcs(t0 + dt))
            a = dt * Kd * (Sd - cd1) * (cl1 + 1)
        """
        eta = self.semi_discrete_field.regularization_factor
        Sd = self.semi_discrete_field.model.Sd
        Kd = self.semi_discrete_field.model.Kd
        cs0 = at_t0.cs
        cl0 = at_t0.cl
        cd0 = at_t0.cd
        xx, yy = self._grid.xx, self._grid.yy
        t1 = t0 + dt
        fcs0 = self.semi_discrete_field.fcs(t0, xx, yy)
        fcs1 = self.semi_discrete_field.fcs(t1, xx, yy)
        RegHCs0 = heaviside_regularized(cs0, eta)

        y = 2*cs0 - dt * Kd * (Sd - cd0) * (cl0 + 1) * RegHCs0 + dt * (fcs0 + fcs1)
        a = dt * Kd * (Sd - cd1) * (cl1 + 1)

        cs1 = self._newton_iterations(y, a, eta, cs0)
        return cs1 * self._grid.null_bd_mask

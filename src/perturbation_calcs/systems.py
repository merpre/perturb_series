# %%
from functools import cached_property
from typing import Union, Tuple
import numpy as np
import scipy as sp

"""Two exemplary systems for perturbation calculations:
- TwoLevelSystem: Simple two-level system with degenerate perturbation
- BoseHubbardTrimer: Bose-Hubbard trimer with tunneling and 
                    on-site interaction terms
© 2025 Meret Preuß <meret.preuss@uol.de>"""


class TwoLevelSystem:

    def __init__(self, vals: np.ndarray, perturb: np.ndarray):
        if vals[0] == vals[1]:
            raise ValueError(
                "Degenerate energy levels cannot be treated with non-degenerate perturbation theory."
            )
        self.vals_H0 = vals.copy()
        self.vecs_H0 = np.eye(len(vals))
        self.H0 = np.diag(vals)
        self.V = perturb.copy()
        self.H = self.H0 + self.V

    def get_eigen(
        self, vals_only=False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return eigenvalues, eigenvectors of Hamiltonian"""
        return sp.linalg.eigh(self.H, eigvals_only=vals_only)


class BoseHubbardTrimer:
    """Model: Bose-Hubbard-Trimer
    - hamiltonian consisting of tunneling (proportional to -omega/kappa,
        considered as perturbation) and on-site interaction  terms.
    - fixed particle number N, Fock-states specified
      by occupation Numbers of well 1 and 2: j,k (well 3 is N-j-k)
    - coupling_well_1_3: 0 for linear trimer, 1 for triangular trimer,
      values in between also possible
    - index calculations are for converting a state with
      occupation numbers j,k to a single index
    """

    def __init__(
        self,
        N: int,
        ok: float,
        coupling_well_1_3: int = 1,
        mu: float = 0,
    ) -> None:
        self.N = N
        self.ok = ok  # omega/kappa
        self.coupling_well_1_3 = coupling_well_1_3  # 0 for linear, 1 for triangular,
        # values in between also possible
        self.mu = mu  # chemical potential
        self.dim_H = int(self.N * (self.N + 3) / 2 + 1)

    def calc_idx(self, j: int, k: int):
        """Convert occupation numbers j,k to single index.
        Taken from Chefles, A. (1996). Nearest-neighbour level spacings 
        for the non-periodic discrete nonlinear Schr¨odinger equation. 
        Journal of Physics A: Mathematical
        and General, 29(15):4515."""
        return j * (self.N - (j - 3) / 2) + k

    def invert_idx(self, idx: int) -> np.ndarray:
        """Convert single index to occupation numbers j,k
        Concluded from Chefles, A. (1996). Nearest-neighbour level spacings 
        for the non-periodic discrete nonlinear Schr¨odinger equation. 
        Journal of Physics A: Mathematical
        and General, 29(15):4515."""
        minus_p_half = (2 * self.N + 3) / 2
        k = 0
        not_found = True
        j = -1
        while k < self.N + 1 and not_found:
            if 2 * k > -(minus_p_half**2) + 2 * idx:
                j = minus_p_half - np.sqrt(minus_p_half**2 - 2 * idx + 2 * k)
                if j % 1 == 0:
                    not_found = False
            k += 1
        return np.array([int(j), k - 1])

    @cached_property
    def j_k_for_idx(self) -> np.ndarray:
        """Calculate all occupation numbers j,k for all indices"""
        indices = np.zeros((self.dim_H, 3))
        for i in range(self.dim_H):
            jk = self.invert_idx(i)
            j = jk[0]
            k = jk[1]
            third = self.N - j - k
            indices[i, :] = np.array([j, k, third])
        return indices

    def calc_hamiltonian_element(self, row: int, column: int) -> float:
        """Calculate element of Hamiltonian matrix"""
        return (
            self.calc_kinetic_element(row, column)
            + self.calc_potential_element(row, column)
            - self.mu * self.N
        )

    def calc_kinetic_element(self, row: int, column: int) -> float:
        """Calculate element of kinetic part of Hamiltonian matrix (omega-dependent)"""
        row_jk = self.invert_idx(row)
        column_jk = self.invert_idx(column)
        row_j = row_jk[0]  # j'
        row_k = row_jk[1]  # k'
        column_j = column_jk[0]  # j
        column_k = column_jk[1]  # k
        tunneling_1_2 = (
            np.sqrt(column_k * (column_j + 1)) * (row_j == column_j + 1)
        ) * ((row_k) == (column_k - 1)) + (
            np.sqrt((column_k + 1) * column_j) * (row_j == column_j - 1)
        ) * (
            (row_k) == (column_k + 1)
        )
        tunneling_2_3 = (row_j == column_j) * (
            np.sqrt((column_k + 1) * (self.N - column_j - column_k))
            * ((row_k) == (column_k + 1))
            + np.sqrt(column_k * (self.N - column_j - column_k + 1))
            * ((row_k) == (column_k - 1))
        )
        tunneling_1_3 = self.coupling_well_1_3 * (
            (row_k == column_k)
            * (
                np.sqrt((column_j + 1) * (self.N - column_j - column_k))
                * ((row_j) == (column_j + 1))
                + np.sqrt(column_j * (self.N - column_j - column_k + 1))
                * ((row_j) == (column_j - 1))
            )
        )
        result = tunneling_1_2 + tunneling_2_3 + tunneling_1_3
        return -self.ok * result

    def calc_potential_element(self, row: int, column: int) -> float:
        """Calculate element of potential part of Hamiltonian matrix (kappa-dependent)"""
        row_jk = self.invert_idx(row)
        column_jk = self.invert_idx(column)
        row_j = row_jk[0]  # j'
        row_k = row_jk[1]  # k'
        column_j = column_jk[0]  # j
        column_k = column_jk[1]  # k
        result = (
            2 * (row_j**2 + row_k**2)
            + self.N**2
            + 2 * row_j * row_k
            - self.N
            - 2 * self.N * (row_j + row_k)
        ) * ((row_j == column_j) and (row_k == column_k))
        return 1 / 2 * result

    @cached_property
    def hamiltonian(self) -> np.ndarray:
        """Calculate complete Hamiltonian matrix"""
        hamiltonian = np.ndarray((self.dim_H, self.dim_H))
        for row in range(self.dim_H):
            for column in range(self.dim_H):
                hamiltonian[row, column] = self.calc_hamiltonian_element(row, column)
        return hamiltonian

    @cached_property
    def kinetic_hamiltonian(self) -> np.ndarray:
        """Calculate kinetic part of Hamiltonian matrix"""
        kinetic_hamiltonian = np.ndarray((self.dim_H, self.dim_H))
        for row in range(self.dim_H):
            for column in range(self.dim_H):
                kinetic_hamiltonian[row, column] = self.calc_kinetic_element(
                    row, column
                )
        return kinetic_hamiltonian

    @cached_property
    def perturbation(self) -> np.ndarray:
        """Calculate kinetic part of Hamiltonian matrix (used as perturbation). Does not contain Omega/kappa!"""
        kinetic_hamiltonian = np.ndarray((self.dim_H, self.dim_H))
        for row in range(self.dim_H):
            for column in range(self.dim_H):
                kinetic_hamiltonian[row, column] = self.calc_kinetic_element(
                    row, column
                )
        return kinetic_hamiltonian / self.ok

    def potential_hamiltonian(self) -> np.ndarray:
        """Calculate potential part of Hamiltonian matrix"""
        return np.eye(self.dim_H) * np.diag(self.hamiltonian)

    def get_eigen(
        self, vals_only: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return eigenvalues, eigenvectors of Hamiltonian"""
        return sp.linalg.eigh(self.hamiltonian, eigvals_only=vals_only)

    @cached_property
    def hopping_matrices(self) -> dict:
        """Calculate hopping matrices for all possible transitions. Does not contain Omega/kappa !
        Not yet used in code."""
        hopping_matrices = {}
        # destroy in 1, create in 2
        hopping_matrices[(1, 2)] = np.zeros((self.dim_H, self.dim_H))
        # destroy in 2, create in 1
        hopping_matrices[(2, 1)] = np.zeros((self.dim_H, self.dim_H))
        # destroy in 1, create in 3
        hopping_matrices[(1, 3)] = np.zeros((self.dim_H, self.dim_H))
        # destroy in 3, create in 1
        hopping_matrices[(3, 1)] = np.zeros((self.dim_H, self.dim_H))
        # destroy in 2, create in 3
        hopping_matrices[(2, 3)] = np.zeros((self.dim_H, self.dim_H))
        # destroy in 3, create in 2
        hopping_matrices[(3, 2)] = np.zeros((self.dim_H, self.dim_H))

        for row in range(self.dim_H):
            for column in range(self.dim_H):
                row_jk = self.invert_idx(row)
                column_jk = self.invert_idx(column)
                row_j = row_jk[0]  # j'
                row_k = row_jk[1]  # k'
                column_j = column_jk[0]  # j
                column_k = column_jk[1]  # k
                # destroy in 1, create in 2
                hopping_matrices[(1, 2)][row, column] = (
                    -np.sqrt((column_k + 1) * column_j)
                    * (row_j == column_j - 1)
                    * ((row_k) == (column_k + 1))
                )
                # destroy in 2, create in 1
                hopping_matrices[(2, 1)][row, column] = (
                    -np.sqrt(column_k * (column_j + 1))
                    * (row_j == column_j + 1)
                    * ((row_k) == (column_k - 1))
                )
                # destroy in 1, create in 3
                hopping_matrices[(1, 3)][row, column] = (
                    -self.coupling_well_1_3
                    * (row_k == column_k)
                    * np.sqrt(column_j * (self.N - column_j - column_k + 1))
                    * ((row_j) == (column_j - 1))
                )
                # destroy in 3, create in 1
                hopping_matrices[(3, 1)][row, column] = (
                    -self.coupling_well_1_3
                    * (row_k == column_k)
                    * np.sqrt((column_j + 1) * (self.N - column_j - column_k))
                    * ((row_j) == (column_j + 1))
                )
                # destroy in 2, create in 3
                hopping_matrices[(2, 3)][row, column] = (
                    -(row_j == column_j)
                    * np.sqrt(column_k * (self.N - column_j - column_k + 1))
                    * ((row_k) == (column_k - 1))
                )
                # destroy in 3, create in 2
                hopping_matrices[(3, 2)][row, column] = (
                    -(row_j == column_j)
                    * np.sqrt((column_k + 1) * (self.N - column_j - column_k))
                    * ((row_k) == (column_k + 1))
                )

        return hopping_matrices

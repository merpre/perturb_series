# %%
from functools import cached_property
import numpy as np
from perturbation_calcs.abstract_kato import AbstractSeries

""" 
© 2025 Meret Preuß <meret.preuss@uol.de>>

Class that applies the abstract kato series to a specific system. 
Includes the possibility to use a pre-calculated abstract series (can save computational 
time in case of higher orders). 
Otherwise, the abstract series is calculated upon initialization."""

class SystemSeries(AbstractSeries):
    """ This class represents a perturbational series 
    for a specified system based on a Hamiltonian and a perturbation. 
    Here, the abstract terms of the are system-specifically calculated.
    The series is calculated up to a specified maximum order.
    
    Args:
        max_order (int): The maximum order of the Kato series to calculate.
        vecs_H0 (np.ndarray): The eigenvectors of the Hamiltonian of the unperturbed system.
        vals_H0 (np.ndarray): The eigenvalues of the Hamiltonian of the unperturbed system.
        perturb (np.ndarray): The perturbation applied to the system.
        energy_idx (int): The index of the energy level to calculate the series for.
        lambda_p (float): The perturbation parameter.
        abstract_results (Optional[dict]): A dictionary containing pre-calculated abstract results 
            for the Kato series. If provided, the calculations will be based on these results 
            instead of calculating them from scratch.
        first_order_zero (bool, optional): Whether to exclude the first order correction 
        from the series. Defaults to False.
    """

    def __init__(
        self,
        max_order: int,
        vecs_H0: np.ndarray,
        vals_H0: np.ndarray,
        perturb: np.ndarray,
        energy_idx: int,
        lambda_p: float,
        abstract_results=None,
        first_order_zero: bool = False,
    ) -> None:
        # Check if abstract results are provided
        if not abstract_results:
            super().__init__(
                max_order,
                return_without_first_order=first_order_zero,
            )
            self.abstract_results = self.return_all_orders()
            # clean up the attributes
            del (
                self.first_call,
                self.length_cache,
                self.config_cache,
                self.direct_dmes,
                self.indirect_dmes,
                self.weights_indirect_dmes,
                self.return_without_first_order,
                self.dme_all_orders,
                self.treated_dmes,
                self.dict_eme_labels,
            )
        else:
            # If abstract results are provided, use them
            self.abstract_results = abstract_results.copy()

        self.max_order = max_order
        self.vecs_H0 = vecs_H0.copy()
        self.vals_H0 = vals_H0.copy()
        self.perturb = perturb.copy()
        self.state_idx = np.argsort(self.vals_H0)[energy_idx]
        self.lambda_p = lambda_p
        self.first_order_zero = first_order_zero
        self.E0 = self.vals_H0[self.state_idx]

        # Initialize dictionaries needed for the calculations
        self.s_operator_dict = {}
        self.eme_value_dict = {}
        self.corrections = []
        self.dict_eme_labels = self.abstract_results["eme labels"]
        self.dme_value_dict = {}
        self.series_calculated = False

    def _calculate_eme(self, eme) -> float:
        """Calculate the value of an EME"""
        # Check if s_operator_dict needs to be filled
        if not self.s_operator_dict:
            self._calculate_s_operator_dict()
        inside_bra_ket = self.perturb.copy()
        if eme != []:
            for i in range(1, len(eme) + 1):
                # Calculate process chain
                inside_bra_ket = np.matmul(
                    self.s_operator_dict[eme[-i]].copy(),
                    inside_bra_ket.copy(),
                )
                inside_bra_ket = np.matmul(
                    self.perturb.copy(),
                    inside_bra_ket.copy(),
                )
        return inside_bra_ket[self.state_idx, self.state_idx]

    def _calculate_s_operator_dict(self) -> None:
        """Calculate the S^k operators for all needed k and store them in a dict"""
        for k in range(self.max_order):
            sk = self._calculate_sk(k)
            self.s_operator_dict[k] = sk

    def _calculate_sk(self, k) -> np.ndarray:
        """Calculate the value of S^k"""
        dim_H = len(self.vals_H0)
        idxs_without_state = [j for j in range(dim_H) if j != self.state_idx]
        sum_ = np.zeros((dim_H, dim_H))
        if k == 0:
            sum_[self.state_idx, self.state_idx] = (
                -1
            )
            return sum_
        for i in idxs_without_state:
            projector = np.zeros((dim_H, dim_H))
            projector[i, i] = 1
            sum_ += projector / pow(
                (self.vals_H0[self.state_idx] - self.vals_H0[i]),
                k,
            )
        return sum_

    def _calculate_eme_dict(self) -> None:
        """Go through all emes in the dict (label:eme) and calculate their values"""
        for (
            label,
            eme,
        ) in self.dict_eme_labels.items():
            self.eme_value_dict[label] = self._calculate_eme(eme)

    def calculate_dme_value(self, dme) -> float:
        """Calculate the value of a DME"""
        if not self.eme_value_dict:
            self._calculate_eme_dict()
        dme_value = 1
        for label, occupation in dme.items():
            dme_value *= pow(
                self.eme_value_dict[label],
                occupation,
            )
        return dme_value

    @cached_property
    def calculate_kato_series(self) -> list:
        """Calculate the whole Kato series for the given system"""
        energy_new = self.E0
        if self.first_order_zero:
            abstract_series = self.abstract_results["no first order"].copy()
        else:
            abstract_series = self.abstract_results["full series"].copy()
        if not self.eme_value_dict:
            self._calculate_eme_dict()

        for n in range(1, self.max_order + 1):
            single_order_sum = 0
            for _, dme_dict in abstract_series[n].items():
                dme_value = dme_dict["weight"]
                for label, occupation in dme_dict["EMEs"].items():
                    dme_value *= pow(
                        self.eme_value_dict[label],
                        occupation,
                    )
                single_order_sum += dme_value
            energy_new += single_order_sum * pow(self.lambda_p, n)
            self.corrections.append(
                {
                    "n": n,
                    "DeltaE": single_order_sum * pow(self.lambda_p, n),
                    "E^n": energy_new,
                }
            )
        self.series_calculated = True
        return self.corrections

    def print_kato_results(self):
        """If already calculated, print the calculated Kato series"""
        if self.corrections:
            for i in range(self.max_order):
                print(self.corrections[i])
        else:
            print(
                "System series not calculated yet. Call calculate_kato_series() first."
            )

    def calc_kato_bose_hubbard_nn_coupling(self, n=0):
        """Explicitly calculate the coefficients for ground state calculation of
        the ring-shaped Bose-Hubbard model. Here, additionally to the first order vanishing, 
        the spectral distance between the groud state
        and the rest of the spectrum ist -1. This allows for a structural simplification."""

        if not self.eme_value_dict: # Calculate EMEs if not done yet
            self._calculate_eme_dict()
        if not n:
            n = self.max_order
        if n > 9:
            raise ValueError("This function only works for orders up to 9")
        # The following calculations are based on analystical calculations,
        # see Preuß (2025, unpublished, arXiv link soon to follow) for details.
        needed_emes = [
            [1],
            [1, 1],
            [1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 2, 1],
            [1, 1, 2, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 3, 1],
            [1, 1, 1, 2, 1],
            [1, 1, 2, 1],
            [1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 3, 1],
            [1, 2, 2, 1],
            [1, 1, 1, 1, 2, 1],
            [1, 1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
        vals = {}

        for eme in needed_emes:
            label = self._label_eme(eme)
            value = self.eme_value_dict[label]
            vals[tuple(eme)] = value

        coeffs = {}
        coeffs[1] = 0
        coeffs[2] = vals[(1,)]
        coeffs[3] = vals[(1, 1)]
        coeffs[4] = vals[(1,)] * vals[(1,)] + vals[(1, 1, 1)]
        coeffs[5] = 3 * vals[(1,)] * vals[(1, 1)] + vals[(1, 1, 1, 1)]
        coeffs[6] = (
            2 * pow(vals[(1,)], 3)
            + 3 * vals[(1,)] * vals[(1, 1, 1)]
            + vals[(1, 1, 1, 1, 1)]
            - vals[(1,)] * vals[(1, 2, 1)]
            + 2 * pow(vals[(1, 1)], 2)
        )
        coeffs[7] = (
            11 * pow(vals[(1,)], 2) * vals[(1, 1)]
            + 3 * vals[(1,)] * vals[(1, 1, 1, 1)]
            + 4 * vals[(1, 1)] * vals[(1, 1, 1)]
            - 2 * vals[(1,)] * vals[(1, 1, 2, 1)]
            - 1 * vals[(1, 1)] * vals[(1, 2, 1)]
            + 1 * vals[(1, 1, 1, 1, 1, 1)]
        )

        coeffs[8] = (
            5 * pow(vals[(1,)], 4)
            + 11 * pow(vals[(1,)], 2) * vals[(1, 1, 1)]
            - 6 * pow(vals[(1,)], 2) * vals[(1, 2, 1)]
            + 1 * pow(vals[(1,)], 2) * vals[(1, 3, 1)]
            + 17 * vals[(1,)] * pow(vals[(1, 1)], 2)
            - 2 * vals[(1,)] * vals[(1, 1, 1, 2, 1)]
            - 1 * vals[(1,)] * vals[(1, 1, 2, 1, 1)]
            + 4 * vals[(1, 1)] * vals[(1, 1, 1, 1)]
            - 2 * vals[(1, 1)] * vals[(1, 1, 2, 1)]
            + 2 * pow(vals[(1, 1, 1)], 2)
            - 1 * vals[(1, 1, 1)] * vals[(1, 2, 1)]
            + 3 * vals[(1,)] * vals[(1, 1, 1, 1, 1)]
            + 1 * vals[(1, 1, 1, 1, 1, 1, 1)]
        )

        coeffs[9] = (
            40 * pow(vals[(1,)], 3) * vals[(1, 1)]
            + 11 * pow(vals[(1,)], 2) * vals[(1, 1, 1, 1)]
            - 12 * pow(vals[(1,)], 2) * vals[(1, 1, 2, 1)]
            + 2 * pow(vals[(1,)], 2) * vals[(1, 1, 3, 1)]
            + 2 * pow(vals[(1,)], 2) * vals[(1, 2, 2, 1)]
            + 34 * vals[(1,)] * vals[(1, 1)] * vals[(1, 1, 1)]
            - 14 * vals[(1,)] * vals[(1, 1)] * vals[(1, 2, 1)]
            + 2 * vals[(1,)] * vals[(1, 1)] * vals[(1, 3, 1)]
            + 3 * vals[(1,)] * vals[(1, 1, 1, 1, 1, 1)]
            - 2 * vals[(1,)] * vals[(1, 1, 1, 1, 2, 1)]
            - 2 * vals[(1,)] * vals[(1, 1, 1, 2, 1, 1)]
            + 8 * pow(vals[(1, 1)], 3)
            + 2 * vals[(1, 1)] * vals[(1, 1, 1, 1, 1)]
            - 2 * vals[(1, 1)] * vals[(1, 1, 1, 2, 1)]
            - 1 * vals[(1, 1)] * vals[(1, 1, 2, 1, 1)]
            + 4 * vals[(1, 1, 1)] * vals[(1, 1, 1, 1)]
            - 2 * vals[(1, 1, 1)] * vals[(1, 1, 2, 1)]
            - 1 * vals[(1, 1, 1, 1)] * vals[(1, 2, 1)]
            + 2 * vals[(1, 1)] * vals[(1, 1, 1, 1, 1)]
            + 1 * vals[(1, 1, 1, 1, 1, 1, 1, 1)]
        )
        powers_lambda = [pow(self.lambda_p, i) for i in range(1, n + 1)]
        coeffs_with_lambda = []

        for i in range(1, n + 1):
            coeffs_with_lambda.append(coeffs[i] * powers_lambda[i - 1])
        return coeffs, coeffs_with_lambda

    def conv_radius(self, bose_hubbard_ring=False, bose_hubbard_N=0):
        """Calculate the convergence radius of the Kato series based 
        on the convergence criterion given by T.Kato (1949). 
        Contains the possibility to calculate the convergence radius
        specified for the ring-shaped Bose-Hubbard system."""
        if not bose_hubbard_N:
            raise ValueError("Please provide the number of particles N.")
        if bose_hubbard_ring:
            dist_spectrum = 1
            pert_norm = 2 * bose_hubbard_N
        else:
            dist_spectrum = sorted(np.abs(self.vals_H0 - self.E0))[1]
            # use the spectral norm of the perturbation
            pert_norm = np.linalg.norm(self.perturb, ord=2)
        lambda_c = dist_spectrum / (2 * pert_norm)

        return lambda_c

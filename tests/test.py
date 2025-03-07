
"""© 2025 Meret Preuß <meret.preuss@uol.de>"""

import unittest
import numpy as np
import scipy as sp
from perturbation_calcs.abstract_kato import AbstractSeries
from perturbation_calcs.system_kato import SystemSeries
from perturbation_calcs.systems import TwoLevelSystem, BoseHubbardTrimer

class TestAbstractSeries(unittest.TestCase):
    def setUp(self):
        max_order = 8
        self.abstract_series = AbstractSeries(
            max_order, return_without_first_order=True
        )
        self.results = self.abstract_series.return_all_orders()

    def test_number_of_nonzero_contributions(self):
        """Test, if the correct number of terms is generated"""
        # from Teichmann et al. 2009
        required_number_of_terms = [
            1,
            1,
            2,
            4,
            10,
            22,
            53,
            119,
            278,
            627,
            1433,
            3216,
            7253,
            16169,
            36062,
            79876,
            176668,
            388910,
            854493,
        ]
        calc_number_of_terms = []
        for n in range(1, len(self.results["full series"]) + 1):
            calc_number_of_terms.append(len(self.results["full series"][n]))
        self.assertEqual(
            calc_number_of_terms, required_number_of_terms[: len(calc_number_of_terms)]
        )

    def test_number_of_reduced_contributions(self):
        """Test if the correct number of terms is generated if all terms containing the first order correction are excluded"""
        # from Teichmann et al. 2009
        required_number_of_terms = [
            0,
            1,
            1,
            2,
            3,
            7,
            12,
            26,
            47,
            97,
            180,
            357,
            668,
            1297,
            2428,
            4628,
            8637,
            16260,
            30188,
            56252,
            10348,
            191873,
            352204,
            646061,
        ]
        calc_number_of_terms = []
        for n in range(1, len(self.results["no first order"]) + 1):
            calc_number_of_terms.append(len(self.results["no first order"][n]))
        self.assertEqual(
            calc_number_of_terms, required_number_of_terms[: len(calc_number_of_terms)]
        )


class TestSystemSeries(unittest.TestCase):
    def setUp(self):
        self.max_order = 8
        test_dim = 5
        vecs_H0 = np.eye(test_dim)
        vals_H0 = np.arange(test_dim)
        perturb = np.diag(0.1 * np.ones(test_dim - 1), k=1) + np.diag(
            0.1 * np.ones(test_dim - 1), k=-1
        )
        energy_idx = 0
        lambda_p = 1
        first_order_zero = False
        self.system_series = SystemSeries(
            self.max_order,
            vecs_H0=vecs_H0,
            vals_H0=vals_H0,
            perturb=perturb,
            energy_idx=energy_idx,
            lambda_p=lambda_p,
            first_order_zero=first_order_zero,
        )
        self.system_series_corrections = self.system_series.calculate_kato_series
        self.abstract_series = AbstractSeries(
            max_order=self.max_order, return_without_first_order=True
        )
        self.abstract_results = self.abstract_series.return_all_orders()
        self.system_series_ext = SystemSeries(
            self.max_order,
            vecs_H0=vecs_H0,
            vals_H0=vals_H0,
            perturb=perturb,
            energy_idx=energy_idx,
            lambda_p=lambda_p,
            abstract_results=self.abstract_results,
            first_order_zero=first_order_zero,
        )
        self.system_series_corrections_ext = (
            self.system_series_ext.calculate_kato_series
        )

    def test_check_length_corrections_ext(self):
        """Test if corrections are calculated in each order for the series with an abstract-series input."""
        self.assertEqual(len(self.system_series_corrections_ext), self.max_order)

    def test_check_length_corrections(self):
        """Test if corrections are calculated in each order for the series in the case where SystemSeries generates the AbstractSeries itself."""
        self.assertEqual(len(self.system_series_corrections), self.max_order)

    def test_first_order_correction_zero(self):
        if np.array_equal(np.diag(self.system_series.perturb), np.zeros(self.system_series.vals_H0.shape)):
            self.assertAlmostEqual(self.system_series_corrections[0]["DeltaE"], 0)

class TestSystems(unittest.TestCase):
    def setUp(self):
        vals = np.array([1, 2])
        perturb = np.array([[0, 0.1], [0.1, 0.0]])
        self.two_level_system = TwoLevelSystem(vals, perturb)
        self.two_level_system_series = SystemSeries(
            8,
            vecs_H0=self.two_level_system.vecs_H0,
            vals_H0=vals,
            perturb=perturb.copy(),
            energy_idx=0,
            lambda_p=1,
            first_order_zero=False,
        )
        self.two_level_system_series_corrections = self.two_level_system_series.calculate_kato_series

        N = 9
        omega = 0.001
        kappa = 1
        coupling_well_1_3 = 1
        self.triangular_bht = BoseHubbardTrimer(N, omega, kappa, coupling_well_1_3)
        self.triangular_bht_series = SystemSeries(8, 
            vecs_H0= np.eye(self.triangular_bht.dim_H),
            vals_H0= np.diag(self.triangular_bht.hamiltonian),
            perturb= self.triangular_bht.kinetic_hamiltonian,
            energy_idx=0,
            lambda_p=1,
            first_order_zero=False,
        )
        self.triangular_bht_series_corrections = self.triangular_bht_series.calculate_kato_series

    def test_two_level_system_no_first_order(self):
        self.assertAlmostEqual(self.two_level_system.get_eigen(vals_only=True)[0], self.two_level_system_series_corrections[-1]["E^n"] )

    def test_triangular_bose_hubbard_trimer_hamiltonian(self):
        self.assertTrue(sp.linalg.ishermitian(self.triangular_bht.hamiltonian))
        
    def test_triangular_bose_hubbard_trimer(self):
        vals = self.triangular_bht.get_eigen(vals_only=True)
        self.assertAlmostEqual(vals[0], self.triangular_bht_series_corrections[-1]["E^n"] )


"""
© 2025 Meret Preuß <meret.preuss@uol.de>>

This script calculates and plots the assumed convergence behavior of
the Kato series for a Bose-Hubbard trimer system with 2 different sizes.
It contains the possibility to plot relative errors of the Kato series,
and to use LaTeX formatting for the plot.

The script performs the following steps:
1. Imports necessary libraries and modules.
2. Defines the parameters and variables for the calculation.
3. Calculates the convergence radii for the trimer systems.
4. Initializes arrays for storing the Kato series results.
5. Calculates the Kato series for different values of omega/kappa.
6. Optionally, normalizes the Kato series results.
7. Plots the Kato series results, including the estimated radius of convergence.
8. Saves the plot as an EPS file.

Note: The script assumes the existence of the following modules:
perturbation_calcs.systems,
perturbation_calcs.system_kato.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from perturbation_calcs.systems import BoseHubbardTrimer
from perturbation_calcs.system_kato import SystemSeries



N = np.array([9, 27])
NUM_OMEGA_KAPPA = 20
NUM_OMEGA_KAPPA += 1

ENERGY_IDX = 0
MAX_ORDER = 9
RELATIVE = False  # plot relative errors
TEX_FORMAT = True  # use LaTeX formatting for the plot
USE_FULL_KATO_SERIES = False  # use full Kato series or the coeffcients
# obtained by structural simplification.
# Both return the same result.
system1 = BoseHubbardTrimer(N[0], 1)
system2 = BoseHubbardTrimer(N[1], 1)
system_series1 = SystemSeries(
    max_order=MAX_ORDER,
    vecs_H0=np.eye(system1.dim_H),
    vals_H0=np.diag(system1.hamiltonian),
    perturb=system1.perturbation,
    energy_idx=0,
    lambda_p=1,
    first_order_zero=True,
)
system_series2 = SystemSeries(
    max_order=MAX_ORDER,
    vecs_H0=np.eye(system2.dim_H),
    vals_H0=np.diag(system2.hamiltonian),
    perturb=system2.perturbation,
    energy_idx=0,
    lambda_p=1,
    first_order_zero=True,
)
print("calculating convergence radii")
convergence_radii = [
    system_series1.conv_radius(bose_hubbard_ring=True, bose_hubbard_N=N[0]),
    system_series2.conv_radius(bose_hubbard_ring=True, bose_hubbard_N=N[1]),
]
# Select the maximum convergence radius and round it to the next 0.01,
# adapt the array so the plot contains the convergence radius and some additional values.
max_omegakappa = max(convergence_radii) * 1.2
max_omegakappa = np.ceil(max_omegakappa / 0.01) * 0.01
OMEGA_KAPPA = np.linspace(0, max_omegakappa, NUM_OMEGA_KAPPA)

kato_results1 = np.zeros((len(OMEGA_KAPPA), MAX_ORDER))
kato_results2 = np.zeros((len(OMEGA_KAPPA), MAX_ORDER))
# Analytically known unperturbed ground state energy
E_0 = 1 / 2 * N * (N / 3 - 1)

# Calculate Kato series:

if not USE_FULL_KATO_SERIES:
    print("calculating coeffs")
    coeffs1, _ = system_series1.calc_kato_bose_hubbard_nn_coupling()
    coeffs2, _ = system_series2.calc_kato_bose_hubbard_nn_coupling()
energies_d1 = np.zeros((len(OMEGA_KAPPA)))
energies_d2 = np.zeros((len(OMEGA_KAPPA)))


print("calculating kato series")
for i, omega_kappa in enumerate(OMEGA_KAPPA):
    print(f"Omega/kappa: {omega_kappa}")
    system_o1 = BoseHubbardTrimer(N[0], omega_kappa)
    system_o2 = BoseHubbardTrimer(N[1], omega_kappa)
    energies_d1[i] = system_o1.get_eigen(vals_only=True)[0]
    energies_d2[i] = system_o2.get_eigen(vals_only=True)[0]
    if USE_FULL_KATO_SERIES:
        system_series_o1 = SystemSeries(
            max_order=MAX_ORDER,
            vecs_H0=np.eye(system_o1.dim_H),
            vals_H0=np.diag(system_o1.hamiltonian),
            perturb=system_o1.perturbation,
            energy_idx=0,
            lambda_p=omega_kappa,
            first_order_zero=True,
        )
        system_series_o2 = SystemSeries(
            max_order=MAX_ORDER,
            vecs_H0=np.eye(system_o2.dim_H),
            vals_H0=np.diag(system_o2.hamiltonian),
            perturb=system_o2.perturbation,
            energy_idx=0,
            lambda_p=omega_kappa,
            first_order_zero=True,
        )
        kato_results1[i, :] = np.array(
            [
                system_series_o1.calculate_kato_series[n - 1]["E^n"]
                for n in range(1, MAX_ORDER + 1)
            ]
        )
        kato_results2[i, :] = np.array(
            [
                system_series_o2.calculate_kato_series[n - 1]["E^n"]
                for n in range(1, MAX_ORDER + 1)
            ]
        )
    else:
        # Calculate the Kato series using the coefficients obtained by structural simplification.
        coeffs_with_powers1 = [
            coeffs1[coeff] * pow(omega_kappa, n)
            for n, coeff in zip(range(1, MAX_ORDER + 1), coeffs1)
        ]
        coeffs_with_powers2 = [
            coeffs2[coeff] * pow(omega_kappa, n)
            for n, coeff in zip(range(1, MAX_ORDER + 1), coeffs2)
        ]
        kato_results1[i, :] = np.array(
            [sum(coeffs_with_powers1[:n]) + E_0[0] for n in range(1, MAX_ORDER + 1)]
        )
        kato_results2[i, :] = np.array(
            [sum(coeffs_with_powers2[:n]) + E_0[1] for n in range(1, MAX_ORDER + 1)]
        )

if RELATIVE:
    # subtract the exact energies from each row and divide by the exact energies
    kato_results1 = np.abs(kato_results1 - energies_d1[:, np.newaxis]) / np.abs(
        energies_d1[:, np.newaxis]
    )
    kato_results2 = np.abs(kato_results2 - energies_d2[:, np.newaxis]) / np.abs(
        energies_d2[:, np.newaxis]
    )
# %%

print("plotting")
if TEX_FORMAT:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "font.size": 12,
        }
    )
symbols = ["x", "o", "s", "v", "^", "<", ">", "d", "p", "h", "1", "2", "3"]
colors = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
]
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i in range(MAX_ORDER - 1):
    ax[0].scatter(
        OMEGA_KAPPA,
        kato_results1[:, i],
        marker=symbols[i],
        color=colors[i],
        alpha=0.7,
        label=f"{i+1}",
    )
    ax[1].scatter(
        OMEGA_KAPPA,
        kato_results2[:, i],
        marker=symbols[i],
        color=colors[i],
        alpha=0.7,
        label=f"{i+1}",
    )

ax[0].plot(
    OMEGA_KAPPA, kato_results1[:, MAX_ORDER - 1], label=f"{MAX_ORDER}", color="black"
)
ax[1].plot(
    OMEGA_KAPPA, kato_results2[:, MAX_ORDER - 1], label=f"{MAX_ORDER}", color="black"
)

# plot vertical dotted line at convergence radius with label up to 2 significant digits
ax[0].axvline(x=convergence_radii[0], color="blue", linestyle="--")
ax[1].axvline(x=convergence_radii[1], color="blue", linestyle="--")
ax[0].text(
    convergence_radii[0],
    ax[0].get_ylim()[1] - 0.15 * (ax[0].get_ylim()[1] - ax[0].get_ylim()[0]),
    f"$\\Omega/\\kappa$ = {convergence_radii[0]:.2g}",
    color="blue",
    ha="center",
    va="top",
    bbox=dict(facecolor="white", edgecolor="blue", boxstyle="round,pad=0.2"),
)
ax[1].text(
    convergence_radii[1],
    ax[1].get_ylim()[1] - 0.15 * (ax[1].get_ylim()[1] - ax[1].get_ylim()[0]),
    f"$\\Omega/\\kappa$ = {convergence_radii[1]:.2g}",
    color="blue",
    ha="center",
    va="top",
    bbox=dict(facecolor="white", edgecolor="blue", boxstyle="round,pad=0.2"),
)


# plot exact energies
if not RELATIVE:
    ax[0].plot(
        OMEGA_KAPPA,
        energies_d1,
        color="black",
        linestyle="--",
        label="$E_d$",
        linewidth=3,
    )
    ax[1].plot(
        OMEGA_KAPPA,
        energies_d2,
        color="black",
        linestyle="--",
        label="$E_d$",
        linewidth=3,
    )


ax[0].set_title(f"$N = {N[0]}$")
ax[1].set_title(f"$N = {N[1]}$")
ax[0].set_xlabel("$\\Omega/\\kappa$")
ax[1].set_xlabel("$\\Omega/\\kappa$")

if RELATIVE:
    ax[0].set_ylabel("$|E^{(n)} - E_d|/|E_d|$")
else:
    ax[0].set_ylabel("$E^{(n)}/\\hbar \\kappa$")
ax[0].legend(ncol=2, loc="best", title="order ($n$)", facecolor="white", framealpha=1)
ax[1].legend(ncol=2, loc="best", title="order ($n$)", facecolor="white", framealpha=1)

# logarithmic y axis
if RELATIVE:
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ylimmax = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
    ylimmmin = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0])
    ax[0].set_ylim([ylimmmin, ylimmax])
    ax[1].set_ylim([ylimmmin, ylimmax])

ax[0].grid(which="both")
ax[1].grid(which="both")

plt.savefig(
    f"plots/convergenceN{N}_{RELATIVE* 'rel'}_{TEX_FORMAT*'tex'}.eps",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.show(block=False)

# %%

"""
"© 2025 Meret Preuß <meret.preuss@uol.de>"
This script calculates and plots the coefficients of a perturbation series for a Bose-Hubbard trimer system.

The script performs the following steps:
1. Imports necessary libraries and modules.
2. Defines the parameters for the system and the perturbation series.
3. Calculates the coefficients of the perturbation series for different values of N.
4. Plots the coefficients against the order of the series for each value of N.

Author: Meret Preuß 
"""

import numpy as np
import matplotlib.pyplot as plt
from perturbation_calcs.systems import BoseHubbardTrimer
from perturbation_calcs.system_kato import SystemSeries

# %%

import numpy as np
import matplotlib.pyplot as plt
from perturbation_calcs.systems import BoseHubbardTrimer
from perturbation_calcs.system_kato import SystemSeries

N = 3 * np.arange(1, 15)
ENERGY_IDX = 0
MAX_ORDER = 9
TEX_FORMAT = True

coefficients = np.zeros((len(N), MAX_ORDER))

for i, n in enumerate(N):
    system = BoseHubbardTrimer(n, 1)
    system_series = SystemSeries(
        max_order=MAX_ORDER,
        vecs_H0=np.eye(system.dim_H),
        vals_H0=np.diag(system.hamiltonian),
        perturb=system.perturbation,
        energy_idx=0,
        lambda_p=1,
        first_order_zero=True,
    )
    print(f"calculating coeffs, N = {n}")
    coeffs, _ = system_series.calc_kato_bose_hubbard_nn_coupling()
    coefficients[i] = [coeffs[coeff] for coeff in range(1, MAX_ORDER + 1)]
# %%

if TEX_FORMAT:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "font.size": 12,
        }
    )
# plot coefficients for each N against the order
symbols = ["x", "o", "s", "v", "^", "<", ">", "d", "p", "h", "1", "2", "3", "4"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

# Plot with value of coefficients
for i, n in enumerate(N):
    ax1.plot(
        range(2, MAX_ORDER + 1),
        coefficients[i][1:],
        ":",
        marker=symbols[i],
        label=f"{n}",
        alpha=1,
        markeredgewidth=1,
    )
ax1.set_xlabel("$n$")
ax1.set_ylabel("$c_n$")
ax1.grid(which="both")
ax1.set_yscale("symlog")
ax1.yaxis.set_ticks(ax1.get_yticks()[::2])  # Keep every other y tick

# Plot with absolute value of coefficients
for i, n in enumerate(N):
    ax2.plot(
        range(2, MAX_ORDER + 1),
        np.abs(coefficients[i][1:]),
        ":",
        marker=symbols[i],
        label=f"{n}",
        alpha=1,
    )
ax2.set_xlabel("$n$")
ax2.set_ylabel("$|c_n|$")
ax2.grid(which="both")
ax2.set_yscale("log")
plt.tight_layout()
ax2.legend(loc="center", bbox_to_anchor=(-0.2, -0.2), ncol=8, title="N")
plt.savefig("plots/coefficients_trimer_N.eps", dpi=300, bbox_inches="tight")
plt.show(block=False)

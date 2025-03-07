# perturb_series

A Python package for generating and computing time-independent non-degenerate perturbation theory as formulated by Tosio Kato (1949). It includes structural simplifications for ring-shaped Bose-Hubbard systems and examples like the Bose-Hubbard trimer and the Two-Level System.
Works as supplementary material for the paper Preuß (2025, submitted) "Simplifying higher-order perturbation theory for ring-shaped Bose-Hubbard systems".

## Documentation

The `/doc` folder contains a PDF with:
- Theoretical basics on time-independent perturbation theory.
- Algorithmic details on computation.
- A guide to the Python implementation.

## Repository Structure

- `doc/` – Theoretical background and implementation guide  
- `src/` – Source code in Python
  - `perturbation_calcs/` – Core calculation classes  
    - `abstract_kato.py` – System-independent perturbation series (AbstractSeries)  
    - `system_kato.py` – System-specific perturbation series (inherits from AbstractSeries)  
    - `systems.py` – Example systems with Hamiltonians and diagonalization
  - `coefficients_trimer.py`and `convergence_trimer.py` for reproducing plots in Preuß (2025, submitted)
- `test/` - some tests 



### File Descriptions

- **`abstract_kato.py`**  
  Defines `AbstractSeries`, a class representing perturbation series independent of specific systems. Outputs structured perturbation series information (see Figure 3 in `doc/documentation.pdf`).

- **`system_kato.py`**  
  Implements `SystemSeries`, which extends `AbstractSeries` for specific systems. Requires:  
  - Eigenvalues and eigenvectors of the unperturbed Hamiltonian  
  - The perturbation 
  - The index of the unperturbed eigenstate  
  - The control parameter

- **`systems.py`**  
  Contains two example systems (Two-Level System and Bose-Hubbard trimer), including their Hamiltonians and diagonalization procedures.

## Usage
- See `coefficients_trimer.py` and `convergence_trimer.py`for examples.

## Requirements
- See `requirements.txt`.

## Citation
If you use this work, please cite it using the following DOI: 10.5281/zenodo.14989444


## Contact
Meret Preuß, meret.preuss@uol.de

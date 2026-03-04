# Discrete Poincaré and Bogovskii Operators

This repository contains the core mathematical operators and numerical experiments for the paper "DISCRETE POINCARÉ AND BOGOVSKIŤ OPERATORS ON COCHAINS AND WHITNEY FORMS" by Johnny Guzmán, Anil N. Hirani, Bingyan Liu, and Pratyush Potu.

It provides a comprehensive Python framework for constructing discrete Poincaré operators directly on cochain complexes and spaces of Whitney forms. Additionally, it includes implementations of a discrete Bogovskii operator that accommodates vanishing boundary trace conditions on star-shaped domains.

## Mathematical Framework

The codebase includes constructive realizations of these operators across various geometric and topological settings:
* Combinatorial (simplicial) discrete Poincaré operators using discrete contraction sequences.
* Geometric (singular) discrete Poincaré operators for Whitney forms.
* L-path contraction operators for domains lacking a global star point.
* Discrete Bogovskii operators enforcing homogeneous boundary conditions.

These implementations are designed to strictly satisfy the discrete homotopy identities, such as $dP + Pd = id$ and $d\mathcal{B} + \mathcal{B}d = id$.

## Repository Structure

* `src/discrete_poincare/`: The core Python package containing the mathematical operators, geometric utilities, topological matrices, and field evaluators.
* `experiments/`: Standalone Python scripts reproducing the numerical experiments detailed in the paper.

## Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
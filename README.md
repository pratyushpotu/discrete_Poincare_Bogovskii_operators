# Discrete Poincaré and Bogovskii Operators

This repository contains implementations of the operators and numerical experiments from the paper "DISCRETE POINCARÉ AND BOGOVSKIŤ OPERATORS ON COCHAINS AND WHITNEY FORMS" by Johnny Guzmán, Anil N. Hirani, Bingyan Liu, and Pratyush Potu.

We provide Python code for constructing discrete Poincaré operators directly on cochains and Whitney forms. Additionally, it includes implementations of a discrete Bogovskii operator that accommodates vanishing boundary trace conditions on star-shaped domains.

## Repository Structure

* `src/discrete_poincare/`: The core Python package containing the mathematical operators, geometric utilities, topological matrices, and field evaluators.
* `experiments/`: Python scripts running numerical experiments.
* `figures/`: Python scripts to reproduce figures in the paper.

## Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash

pip install -r requirements.txt

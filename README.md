# **`README.md`**

# **Optimal Social Protection: A Production-Grade Implementation**

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.07431-b31b1b.svg)](https://arxiv.org/abs/2511.07431)
[![DOI](https://img.shields.io/badge/DOI-10.1257/aer.20251107-gray.svg)](https://www.aeaweb.org/journals/aer)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost)
[![Discipline](https://img.shields.io/badge/Discipline-Actuarial%20Science-00529B)](https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost)
[![Discipline](https://img.shields.io/badge/Discipline-Development%20Economics-00529B)](https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost)
[![Data Source](https://img.shields.io/badge/Data%20Source-Synthetic-003299)](https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost)
[![Core Method](https://img.shields.io/badge/Method-Stochastic%20Optimal%20Control-orange)](https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost)
[![Analysis](https://img.shields.io/badge/Analysis-Monte%20Carlo%20Simulation-red)](https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025577.svg?style=flat&logo=SciPy&logoColor=white)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![PyYAML](https://img.shields.io/badge/PyYAML-gray?style=flat)](https://pyyaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Optimal Cash Transfers and Microinsurance to Reduce Social Protection Costs"** by:

*   Pablo Azcue
*   Corina Constantinescu
*   José Miguel Flores-Contró
*   Nora Muler

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous configuration validation to core computation (via both analytical and Monte Carlo methods), robustness analysis, and final artifact generation.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_complete_study`](#key-callable-run_complete_study)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the stochastic optimal control framework presented in Azcue et al. (2025). The core of this repository is the iPython Notebook `optimal_cash_transfers_microinsurance_reduce_social_protection_cost_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline is designed as a robust and scalable system for determining the cost-minimizing cash transfer policy for a government or NGO, with and without the presence of microinsurance.

The paper's central contribution is to frame social protection policy design as a continuous-time optimal control problem and to solve it using both analytical and numerical methods. This codebase operationalizes the paper's framework, allowing users to:
-   Rigorously define and validate an entire economic scenario via a single `config.yaml` file.
-   Compute the optimal cash transfer policy (the optimal threshold `y*`) that minimizes the expected discounted cost of government interventions.
-   Evaluate the cost of this optimal policy and compare it against key baseline policies (e.g., "inject-to-poverty-line" and "perpetual transfers").
-   Analyze the impact of microinsurance (Proportional, Excess-of-Loss, and Total-Loss) on both the optimal policy and the associated government costs.
-   Run a complete, end-to-end analysis with a single function call.
-   Perform a full suite of sensitivity and robustness analyses to test the model's behavior under different parameterizations.
-   Generate a complete, reproducible audit trail for every analysis run.

## Theoretical Background

The implemented methods are grounded in stochastic processes, optimal control theory, and actuarial science.

**1. Household Capital as a PDMP:**
The household's capital, $X_t$, is modeled as a Piecewise-Deterministic Markov Process (PDMP).
-   **Deterministic Flow:** Between shocks, capital grows according to the ODE:
    $$
    dX_t = r[X_t - x^*]^+ dt
    $$
    where $x^*$ is the poverty line and $r$ is the net growth rate.
-   **Stochastic Jumps:** Catastrophic shocks arrive according to a Poisson process with rate $\lambda$. At a shock time $\tau_i$, the capital is reduced multiplicatively: $X_{\tau_i} = X_{\tau_i^-} \cdot Z_i$, where $Z_i$ is the remaining capital proportion.

**2. Stochastic Optimal Control:**
The government's problem is to choose a cumulative transfer process, $S_t$, to minimize the total expected discounted cost, subject to keeping the household's capital above the poverty line. The value function is:
$$
V(x) = \inf_{\pi} \mathbb{E}_x \left[ \int_{0^-}^\infty e^{-\delta t} dS_t \right]
$$
This problem is characterized by a Hamilton-Jacobi-Bellman (HJB) equation. The paper shows that the optimal policy is a **threshold strategy**: intervene only when capital drops to a specific optimal threshold $y^* \ge x^*$, and inject just enough to restore it to $y^*$.

**3. Solution Methods:**
-   **Analytical Solution:** For the specific case where the shock distribution is Beta($\alpha$, 1) and there is no insurance, the value function $V_y(x)$ can be expressed in closed form using the Gaussian hypergeometric function, ${}_2F_1$.
-   **Monte Carlo Simulation:** For the general case (any shock distribution, with or without insurance), the value function is estimated using Monte Carlo simulation. The value at the threshold, $V_y(y)$, is found by solving a renewal equation:
    $$
    \hat{V}^{\pi_y}(y) \approx \frac{\mathbb{E}[J_y e^{-\delta \tau_y}]}{1 - \mathbb{E}[e^{-\delta \tau_y}]}
    $$
    where $(\tau_y, J_y)$ are the first-passage time and injection amount for paths starting at $y$.

**4. Microinsurance Modeling:**
Microinsurance is modeled as a transformation of the underlying PDMP. It reduces the growth rate ($r \to r^R$) and increases the poverty line ($x^* \to x^{*R}$) due to premium payments, but it favorably alters the shock distribution ($Z \to W$). The premium $p_R$ is calculated using the expected value principle:
$$
p_R = (1 + \gamma)\lambda \mathbb{E}[1 - Z - R(1-Z)]
$$
where $R(\cdot)$ is the retained loss function for the specific policy type.

## Features

The provided iPython Notebook (`optimal_cash_transfers_microinsurance_reduce_social_protection_cost_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The entire pipeline is broken down into 28 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All model parameters, computational settings, and analysis stages are controlled by an external `config.yaml` file.
-   **Dual-Method Engine:** Seamlessly switches between the high-speed analytical closed-form solver and the general-purpose, high-fidelity Monte Carlo engine based on the configuration.
-   **Production-Grade Numerics:** Implements best practices for numerical computation, including robust optimization (Brent's method), accurate numerical integration (`scipy.quad`), stable special function evaluation (`scipy.hyp2f1`), and vectorized operations with `NumPy`.
-   **Comprehensive Validation Suite:** Includes a full suite of validation and verification checks, from initial parameter validation to end-to-end sanity checks on the final results (monotonicity, boundary consistency, reproducibility).
-   **Rigorous Uncertainty Quantification:** The Monte Carlo engine uses the method of batch means with the t-distribution to provide statistically sound confidence intervals for all estimates.
-   **Complete Replication and Robustness:** A single top-level function call can execute the entire study, including a comprehensive suite of sensitivity analyses and convergence diagnostics.
-   **Full Provenance:** The pipeline generates a complete, human-readable JSON manifest for each run, containing all inputs, derived parameters, diagnostics, and results for full reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Setup (Tasks 1-3):** Ingests, validates, and cleanses the `config.yaml` file into a safe, typed, and immutable object.
2.  **Parameter Derivation (Tasks 4-6):** Computes all derived quantities for the base and insured models, and selects the active parameters for the run.
3.  **Stochastic Engine (Tasks 7-8):** Constructs and validates the random number samplers and the core PDMP path simulation kernel.
4.  **Estimators (Tasks 9-13):** Implements the Monte Carlo and closed-form evaluators for the value functions $V_y(x)$ and $C(x)$.
5.  **Optimization (Task 15):** Implements the one-dimensional optimization to find the optimal threshold $y^*$.
6.  **Evaluation (Tasks 14, 16, 24):** Computes the final value function curves for the optimal policy and all baseline comparators ($C(x)$, $D(x)$) across a grid.
7.  **Robustness & Verification (Tasks 17, 18, 21-23, 25, 28):** Executes the full suite of sensitivity analyses, convergence diagnostics, and final validation checks.
8.  **Reporting (Tasks 26-27):** Generates all plots and the final run manifest.

## Core Components (Notebook Structure)

The `optimal_cash_transfers_microinsurance_reduce_social_protection_cost_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 28 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_complete_study`

The project is designed around a single, top-level user-facing interface function:

-   **`run_complete_study`:** This master orchestrator function, located in the final section of the notebook, runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, controlled by the `analysis_stages` block in the configuration file.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `numpy`, `scipy`, `matplotlib`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost.git
    cd optimal_cash_transfers_microinsurance_reduce_social_protection_cost
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install numpy scipy matplotlib pyyaml
    ```

## Input Data Structure

The entire pipeline is controlled by a single `config.yaml` file. The structure of this file is detailed in the notebook and the provided example `config.yaml`. It includes sections for metadata, model parameters, computational settings, and run control.

## Usage

The `optimal_cash_transfers_microinsurance_reduce_social_protection_cost_draft.ipynb` notebook provides a complete, self-contained example. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `run_complete_study` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Define the path to the configuration file.
    CONFIG_PATH = "config.yaml"
    
    # 2. Load the configuration from the YAML file into a Python dictionary.
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        print("--- Configuration loaded successfully. ---")
    except Exception as e:
        print(f"--- ERROR: Failed to load configuration. --- \n{e}")
        # Exit or handle error
    
    # 3. Execute the entire study.
    # The `analysis_stages` block within the config file controls which
    # optional analyses (e.g., convergence diagnostics, plots) are run.
    if 'config' in locals():
        master_results = run_complete_study(full_study_configuration=config)
    
        # 4. Inspect and save final artifacts.
        # For example, save the main plot and the run manifest.
        if "plots" in master_results:
            main_plot = master_results["plots"]["main_value_function_plot"]
            main_plot.savefig("final_value_functions.pdf")
            print("\n--- Main plot saved to 'final_value_functions.pdf' ---")
        
        if "run_manifest_json" in master_results:
            manifest_str = master_results["run_manifest_json"]
            with open("run_manifest.json", "w") as f:
                f.write(manifest_str)
            print("--- Run manifest saved to 'run_manifest.json' ---")
```

## Output Structure

The `run_complete_study` function returns a dictionary containing all generated artifacts. If saved, the primary outputs are:
-   **`run_manifest.json`**: A complete JSON file containing all inputs, derived parameters, diagnostics, and numerical results for the run.
-   **`*.pdf` / `*.png`**: Plot files generated by the visualization stage.

## Project Structure

```
optimal_cash_transfers_microinsurance_reduce_social_protection_cost/
│
├── optimal_cash_transfers_microinsurance_reduce_social_protection_cost_draft.ipynb
├── config.yaml
├── requirements.txt
├── LICENSE
└── README.md
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify all study parameters, including economic assumptions, shock distributions, insurance policies, and computational settings, without altering the core Python code. New shock distributions or insurance types can be added by extending the relevant helper functions (e.g., `_create_z_sampler`, `_compute_insurance_premium`).

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

-   **Multi-dimensional Sweeps:** Extend the `run_parameter_sweep` and `plot_sensitivity_sweep` functions to handle and visualize 2D parameter sweeps (e.g., generating a heatmap of `y*` as a function of `λ` and `δ`).
-   **Parallelization:** The `run_parameter_sweep` and `_generate_first_passage_samples` functions are embarrassingly parallel. An extension could use `joblib` or `multiprocessing` to significantly accelerate large-scale analyses.
-   **Additional Distributions:** Add support for other shock distributions (e.g., Lognormal, Pareto) by implementing their samplers and required properties (PDF, CDF, mean).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{azcue2025optimal,
  title={Optimal Cash Transfers and Microinsurance to Reduce Social Protection Costs},
  author={Azcue, Pablo and Constantinescu, Corina and Flores-Contr{\'o}, Jos{\'e} Miguel and Muler, Nora},
  journal={arXiv preprint arXiv:2511.07431},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Production-Grade Implementation of "Optimal Cash Transfers and Microinsurance to Reduce Social Protection Costs".
GitHub repository: https://github.com/chirindaopensource/optimal_cash_transfers_microinsurance_reduce_social_protection_cost
```

## Acknowledgments

-   Credit to **Pablo Azcue, Corina Constantinescu, José Miguel Flores-Contró, and Nora Muler** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **NumPy, SciPy, and Matplotlib**.

--

*This README was generated based on the structure and content of the `optimal_cash_transfers_microinsurance_reduce_social_protection_cost_draft.ipynb` notebook and follows best practices for research software documentation.*

# Code repository for "Numerical Analysis of a Nonlinear Temperature Enhanced Diffusion Model for Drug Delivery"

This repository contais the code for "Numerical Analysis of a Nonlinear Temperature Enhanced Diffusion Model for Drug Delivery".

To run the code in this repository, first you should set up the `prob1_env` conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate prob1_env
```

Core source files are in `src/`. Convergence studies are in Jupyter Notebook files in the root directory.

## Intended use

Once the conda environment is set up, you can verify that all tests are passing by running:

```bash
pytest tests
```

Then, running the notebooks for reproducing the convergence studies is the next step. To run a notebook, it should be in project's root folder. Simply open the notebook and run all cells. The first cell will add `src/` to the Python's system path.

## Files Structure

Here is a brief description of the purpose of the files in `src/`:

- `prob1base.py` - Core logic, numerical method, discretization, etc.

- `utils_for_testing.py` - Utilities specifically for the test files in `tests/`.

- `cvg_studies_base.py` - Contains helper functions for programming the convergence studies. Not notebook specific. Can be used for regular scripts whose purpose is to perform convergence studies as well.

- `mms_trial_utils.py` - Main utilities for running a simulation against a manufactured solution.

- `prob1_mms_cases.py` - A library of manufactured solutions for studies.

- `notebook_studies_helpers.py` - Similar to `cvg_studies_base.py`, however these target utilities specifically helpful to notebook studies.

The following list is a description of the notebook files, and the convergence studies they cover. Some remarks are in order: "MMS" refers to "Method of Manufactured Solutions", "reghcstriple" refers to "regularized Heaviside applied to $c_s$ within its containing triple product". Not just that, the following descriptions might be cryptic without knowledge from the paper.

- `MMSCaseNonFullySmoothPol_cpcsH2_TclcdH2_reghcstriple_convergence_study` - For this study, we'll use the `MMSCaseNonFullySmoothPol_cpcsH2_TclcdH2` MMS case: a solution which is just short regular enough on the $\mathbb P_1$ variables ($T, c_l, c_d$) for order 2 spatial convergence according to our results, while $\mathbb P_0$ variables ($c_p, c_s$) are just regular enough. More precisely, all $\mathbb P_0$ and $\mathbb P_1$ variables are in $H^2 \setminus H^3$. See `src/prob1_mms_cases.py' for a precise definition. This study expects order 2 spatial convergence breakdown.

- `MMSCaseNonFullySmoothPol_cpcsH1_TclcdH2_reghcstriple_convergence_study` - For this study, we'll use the `MMSCaseNonFullySmoothPol_cpcsH1_TclcdH2` case: a solution which is just short of regular enough for order 2 spatial convergence. More precisely, $\mathbb P_0$ variables ($c_p, c_s$) are in $H^1 \setminus H^2$, and $\mathbb P_1$ variables ($T, c_l, c_d$) are in $H^2 \setminus H^3$. See `src/prob1_mms_cases.py' for a precise definition. This study expects order 2 spatial convergence breakdown.

- `MMSCaseExpSin_reghcstriple_convergence_study` - For this study, we'll use the `MMSCaseExpSin` case: smooth solution with sinusoidal behavior. See `src/prob1_mms_cases.py' for a precise definition. This study expects order 2 spatial and temporal convergence.

- `MMSCaseNonFullySmoothPol_cpcsH2_TclcdH3_reghcstriple_convergence_study` - For this study, we'll use the `MMSCaseNonFullySmoothPol_cpcsH2_TclcdH3` case. This provides a solution which is just regular enough for order 2 spatial convergence according to your results. More specifically, $\mathbb P_1$ variables ($T, c_l, c_d$) are in $H^3 \setminus H^4$ and $\mathbb P_0$ variables ($c_p, c_s$) are in $H^2 \setminus H^3$. See `src/prob1_mms_cases.py' for a precise definition. This study expects order 2 spatial and temporal convergence.

- `MMSCasePol_reghcstriple_convergence_study` - For this study, we'll use the `MMSCasePol` case: a smooth solution with a polynomial spacial profile. See `src/prob1_mms_cases.py' for a precise definition. This study expects order 2 spatial and temporal convergence.

- `MMSCaseSlowlyChangingPeaks_Fast1e1_reghcstriple_convergence_study` - For this study, we'll use the `MMSCaseSlowlyChangingPeaks_Fast1e1` case: a smooth solution designed to lead to larger errors. The objective is to better see the convergence happening. See `src/prob1_mms_cases.py' for a precise definition. This study expects order 2 spatial and temporal convergence.

## Disclaimer

This repository contains the code used to generate results for the associated scientific work and is provided as-is for reproducibility and research purposes.

The code is distributed under the MIT license. See `LICENSE`.
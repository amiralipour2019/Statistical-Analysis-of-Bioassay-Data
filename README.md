# Statistical-Analysis-of-Bioassay-Data
This repository offers an analysis of bioassay data to estimate LD50 using Python. It employs logistic regression, bootstrap resampling, and compares results with posterior distributions for precise parameter estimation and uncertainty assessment.


## Description of the Problem
The bioassay data analysis aims to estimate parameters $\alpha$ and $\beta$ that describe the relationship between dose levels and the probability of an outcome (e.g., death in toxicology studies). The analysis further seeks to estimate the LD50, the dose at which there is a 50% chance of observing the outcome, and assess the uncertainty of these estimates through confidence intervals and comparison with posterior distributions.

## Project Structure
- **Exploratory Data Analysis (EDA)**: Initial visualization and statistical summary to explore the dose-response curve.
- **Modeling**: Application of logistic regression to model the dose-response relationship, including parameter estimation.
- **Bootstrap Analysis**: Use of bootstrap resampling for constructing 95% confidence intervals for \(\alpha\), \(\beta\), and LD50.
- **Delta Method Analysis**: Employing the Delta Method for variance estimation of LD50, offering a nuanced approach to uncertainty quantification.
- **Comparative Analysis**: Evaluating Non-Bayesian estimates against Bayesian posterior distributions to validate methodological robustness.
- **Visualization and Statistical Summary**: Detailed analysis and graphical representation to convey findings effectively.

## Dataset
Includes dose levels, the number of animals tested, and the number of deaths at each dose level from a bioassay experiment.

## Implementation
Implemented in Python, using Pandas, Matplotlib, SciPy, and other libraries for data handling, visualization, and statistical modeling.

```python
# Sample snippet for logistic regression, bootstrap analysis, and Delta Method implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import binom
from sklearn.utils import resample

# Define logistic regression model, bootstrap resampling, and Delta Method functions



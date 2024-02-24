# Statistical-Analysis-of-Bioassay-Data
This repository offers an analysis of bioassay data to estimate LD50 using Python. It employs logistic regression, delta method approximation, bootstrap resampling, and compares results with posterior distributions for precise parameter estimation and uncertainty assessment.


## Description of the Problem
The bioassay data analysis aims to estimate parameters $\alpha$ and $\beta$ that describe the relationship between dose levels and the probability of an outcome (e.g., death in toxicology studies). The analysis further seeks to estimate the LD50, the dose at which there is a 50% chance of observing the outcome, and assess the uncertainty of these estimates through confidence intervals and comparison with posterior distributions.

## Project Structure
- **Exploratory Data Analysis (EDA)**: Initial visualization and statistical summary to explore the dose-response curve.
- **Modeling**: Application of logistic regression to model the dose-response relationship, including parameter estimation.
- **Bootstrap Analysis**: Use of bootstrap resampling for constructing 95% confidence intervals for $\alpha$, $\beta$, and LD50.
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

# Define Bioassay Data
bioassay_data=pd.DataFrame({
    'Dose_Log_g_ml':[-0.86,-0.30,-0.05,0.73],
    'Number_of_Animals':[5,5,5,5],
    'Number_of_Deaths':[0,1,3,5]
})

bioassay_data.head()
bioassay_data.shape
```


## Results
- **Parameter Estimates**: Derived from logistic regression to define the dose-response relationship.
- **LD50 Estimates**: Calculated using model estimates and refined with the Delta Method.
- **Confidence Intervals**: Constructed through bootstrap analysis and Delta Method calculations for parameters and LD50.
## Conclusion
The project demonstrates the efficacy of Non-Bayesian methods, including logistic regression, bootstrap analysis, and the Delta Method, in bioassay data analysis. These approaches provide reliable estimates of LD50, showcasing their value in pharmacology and toxicology research.


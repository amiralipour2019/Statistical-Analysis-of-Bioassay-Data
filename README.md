# Statistical-Analysis-of-Bioassay-Data
This repository offers an analysis of bioassay data to estimate LD50 using Python. It employs logistic regression, bootstrap resampling, and compares results with posterior distributions for precise parameter estimation and uncertainty assessment.


## Description of the Problem
The bioassay data analysis aims to estimate parameters $\alpha$ and $\beta$ that describe the relationship between dose levels and the probability of an outcome (e.g., death in toxicology studies). The analysis further seeks to estimate the LD50, the dose at which there is a 50% chance of observing the outcome, and assess the uncertainty of these estimates through confidence intervals and comparison with posterior distributions.

## Coding
The analysis was performed using Python, leveraging libraries such as NumPy, SciPy, and Scikit-learn. Key steps in the analysis include:

- **Parameter Estimation**: Parameters were estimated using logistic regression and direct logit function application, optimizing the log-likelihood function.
- **Bootstrap Analysis**: Bootstrap resampling techniques were employed to construct 95% confidence intervals for \(\alpha\), \(\beta\), and LD50, providing insights into the estimates' variability.
- **Comparison with Posterior Distributions**: The confidence intervals and parameter estimates were compared with empirical and normal-approximated posterior distributions to evaluate the accuracy and reliability of the inferential summaries.

```python
# Python code snippet for bootstrap analysis
import numpy as np
from sklearn.utils import resample
# Additional code for logistic regression and logit function

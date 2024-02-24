
# call the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


    
# Define Bioassay Data
bioassay_data=pd.DataFrame({
    'Dose_Log_g_ml':[-0.86,-0.30,-0.05,0.73],
    'Number_of_Animals':[5,5,5,5],
    'Number_of_Deaths':[0,1,3,5]
})

bioassay_data.head()
bioassay_data.shape

#Exploratory Data Analysis (EDA):
    
# 1: calculate the mortality rate
bioassay_data['Mortality_Rate']=bioassay_data['Number_of_Deaths']/bioassay_data['Number_of_Animals']

print(bioassay_data)


# 2: # Plotting the dose-response curve
plt.figure(figsize=(12,10))
plt.scatter(bioassay_data['Dose_Log_g_ml'],bioassay_data['Mortality_Rate'],color='blue', label='Observed Data')
plt.plot(bioassay_data['Dose_Log_g_ml'],bioassay_data['Mortality_Rate'],color='red', label='Trend',linestyle='--')
#lt.title('Dose-Response Curve')
plt.xlabel('Dose(log g/ml)')
plt.ylabel('Mortality Rate')
plt.legend()
plt.grid(True)
plt.show()

#3: Checking for any anomalies or outliers by Boxplot
plt.figure(figsize=(12,10))
plt.subplot(1,2,1)
plt.boxplot(bioassay_data['Dose_Log_g_ml'], patch_artist=True) # 'patch_artist=True' fills the box with color
plt.title('Boxplot of Dose Usage')
plt.subplot(1,2,2)
plt.boxplot(bioassay_data['Number_of_Deaths'], patch_artist=True)
plt.title('Bxplot of Number of Deaths')
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # Adjust the space between the plots
plt.show()


# Descriptive Statistics
print(bioassay_data.describe())

########### Compute the Poserior of alpha and beta
from scipy.stats import binom, uniform
from scipy.special import expit as logistic_function

# Step 1: Rough estimates of alpha and beta
alpha_hat,beta_hat= 0.8,7.7 

# Step 2: Define the grid
alpha_range=np.linspace(-5,10,100)
beta_range=np.linspace(-10,40,150)
alpha_grid,beta_grid = np.meshgrid(alpha_range, beta_range,indexing='ij')#using 'ij' indexing for compatibility
# Step 3: Compute the posterior distribution for each (alpha, beta) pair on the grid
posterior=np.zeros(alpha_grid.shape)  # Initialize the posterior matrix with zeros

# Iterate over the grid of alpha and beta values
for i in range(len(alpha_range)):
    for j in range(len(beta_range)):
        alpha=alpha_grid[i,j]  # Current alpha value from the grid
        beta=beta_grid[i,j]   # Current beta value from the grid
        
        # Calculate the probability of death using the logistic function for the given alpha, beta, and doses
        p=logistic_function(alpha+beta*bioassay_data['Dose_Log_g_ml'])
        
        # Compute the likelihood of observing the given number of deaths for each dose, given p
        likelihood=binom.pmf(bioassay_data['Number_of_Deaths'],bioassay_data['Number_of_Animals'],p)
        
        # Multiply the likelihoods across all doses (assuming independence) to get the joint likelihood
        # Assuming a uniform prior, so the posterior is proportional to the likelihood
        posterior[i, j]=np.prod(likelihood)
print(posterior)
# Step 4: Normalize the posterior
posterior /=np.sum(posterior)
print(np.sum(posterior))

# Step 5: Sampling from the posterior
# Simplified for illustration; assumes uniform sampling over grid indices

# Randomly select 1000 indices from the posterior distribution, using the flattened posterior as probabilities
sample_indices= np.random.choice(range(posterior.size),size=1000,p=posterior.ravel())

# Convert the flat indices back to 2D indices for alpha and beta in the grid
sample_alphas,sample_betas = np.unravel_index(sample_indices,posterior.shape)

# Map the indices back to the corresponding alpha values
sample_alphas=alpha_range[sample_alphas]

# Map the indices back to the corresponding beta values
sample_betas=beta_range[sample_betas]

# Step 6: Visualization
# Visualization: Contour plot of the posterior and illustrative scatter plot
plt.figure(figsize=(16, 10))

# Contour plot of the posterior density
plt.subplot(1,2,1)
contour_levels=np.linspace(np.min(posterior),np.max(posterior),50)
plt.contourf(alpha_grid,beta_grid,posterior,levels=contour_levels,cmap='viridis')
plt.colorbar(label='Posterior probability')
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.title('Posterior Density Contour')
# Scatter plot of sampled alpha and beta
plt.subplot(1,2,2)
plt.scatter(sample_alphas,sample_betas,alpha=0.5)
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.title('Sampled Values Scatter Plot')

plt.tight_layout()
plt.show()

# Delta method
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Placeholder values for the demonstration
alpha_hat = 0.8  # Estimated mean of alpha
beta_hat = 7.7   # Estimated mean of beta
sigma_alpha = 1.0  # Standard deviation of alpha
sigma_beta = 2.0   # Standard deviation of beta
sigma_alpha_beta = 0.5  # Covariance between alpha and beta

# Constructing the covariance matrix
Sigma = np.array([[sigma_alpha**2, sigma_alpha_beta], [sigma_alpha_beta, sigma_beta**2]])

# Generate a grid of alpha, beta values for contour plot
x,y=np.mgrid[-1:2:.01,5:10:.01]
pos=np.empty(x.shape+(2,))
pos[:,:,0]=x; pos[:,:,1]=y

# Creating a multivariate normal distribution for alpha and beta
rv=multivariate_normal([alpha_hat, beta_hat], Sigma)

# Generate simulated LD50 values using the delta method for normal approximation
std_dev_ld50 = np.sqrt(np.dot(np.array([-1/beta_hat, alpha_hat/beta_hat**2]).T, np.dot(Sigma, np.array([-1/beta_hat, alpha_hat/beta_hat**2]))))
ld50_delta_method = np.random.normal(-alpha_hat / beta_hat, std_dev_ld50, 1000)



# Plotting
fig,axs =plt.subplots(1,3,figsize=(18, 6))

# Contour plot for joint distribution of alpha and beta
axs[0].contourf(x, y, rv.pdf(pos))
axs[0].set_title('Contour Plot of Alpha and Beta')
axs[0].set_xlabel('Alpha')
axs[0].set_ylabel('Beta')

# Scatter plot for simulated samples of alpha and beta
samples = rv.rvs(1000)
axs[1].scatter(samples[:, 0], samples[:, 1], alpha=0.5)
axs[1].set_title('Scatter Plot of Samples')
axs[1].set_xlabel('Alpha')
axs[1].set_ylabel('Beta')

# Histogram of LD50 values using delta method approximation
axs[2].hist(ld50_delta_method, bins=30, color='lightgreen', edgecolor='black', alpha=0.6, density=True)
axs[2].set_title('Delta Method Approximation of LD50')
axs[2].set_xlabel('LD50')
axs[2].set_ylabel('Density')

plt.tight_layout()
plt.show()

# Summary statistics of Delta method
mean_ld50_delta=np.mean(ld50_delta_method)
median_ld50_delta=np.median(ld50_delta_method)
sdv_ld50_delta=np.std(ld50_delta_method)

print(f"Mean of LD50 delta samples: {mean_ld50_delta}")
print(f"Median of LD50 delta samples: {median_ld50_delta}")
print(f"Standard deviation of LD50 delta samples: {sdv_ld50_delta}")


## Normal approximation
from scipy.stats import multivariate_normal
 # Initialize  optimized parameters and covariance matrix
optimized_params=np.array([0.8,7.7]) 
cov_matrix=np.array([[1, 0.5],[0.5, 2]]) 

# Generating a grid for contour plot
x,y=np.mgrid[-1:2:.01,-10:20:.01]
pos=np.dstack((x, y))
rv=multivariate_normal(optimized_params, cov_matrix)

# Sampling from the multivariate normal distribution
samples=multivariate_normal.rvs(mean=optimized_params,cov=cov_matrix,size=1000)
print(samples.shape)

# Filter samples where beta > 0 and calculate LD50
beta_positive_samples=samples[samples[:,1]>0]
ld50_samples= -beta_positive_samples[:,0]/beta_positive_samples[:,1]

# Plotting
fig,axs=plt.subplots(1,3,figsize=(18,6))

# Contour plot for the posterior density
axs[0].contourf(x,y,rv.pdf(pos))
axs[0].set_title('Contour Plot of Posterior Density')
axs[0].set_xlabel('Alpha')
axs[0].set_ylabel('Beta')

# Scatter plot of alpha and beta samples
axs[1].scatter(samples[:,0],samples[:,1],alpha=0.5)
axs[1].set_title('Scatter Plot of Samples')
axs[1].set_xlabel('Alpha')
axs[1].set_ylabel('Beta')

# Histogram of LD50 values
axs[2].hist(ld50_samples,bins=30,color='steelblue',edgecolor='black')
axs[2].set_title('Histogram of LD50 Values (Beta > 0)')
axs[2].set_xlabel('LD50')
axs[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


#Summary statitics of normal approximation
ld50_mean=ld50_samples.mean()
ld50_median=np.median(ld50_samples)
ld50_std=np.std(ld50_samples)

print(f"Mean of LD50 samples: {ld50_mean}")
print(f"Median of LD50 samples: {ld50_median}")
print(f"Standard deviation of LD50 samples: {ld50_std}")

# Non-Bayesian Analysis of Bioassay Data
# call required libaray
from scipy.optimize import minimize
# Bioassay data
x=np.array([-0.86, -0.30, -0.05, 0.73])
n=np.array([5, 5, 5, 5])
y=np.array([0, 1, 3, 5])

# MLE Estimator of α, β
# define the Logit Function
def logit(theta):
    return np.log(theta/(1-theta))

#define inverse logit function
def inv_logit(x):
    return np.exp(x)/(1+np.exp(x))

# define likelihood function
def neg_log_likelihood(params):
    alpha,beta=params
    theta=inv_logit(alpha+beta*x)
    likelihood=y*np.log(theta)+(n-y)*np.log(1-theta)
    return -np.sum(likelihood)

# Initial guess values for alpha and beta
initial_values=[0,0]

# Minimize the negative log likelihood
res=minimize(neg_log_likelihood,initial_values, method="BFGS")#Broyden–Fletcher–Goldfarb–Shanno algorithm

#get the MLEs
alpha_hat,beta_hat=res.x
alpha_hat,beta_hat
# Estimating the bias and variance of model parameters (α and β)
from scipy.optimize import minimize
from scipy.stats import binom
import numpy as np

# True values for alpha and beta (for simulation purposes)
true_alpha=0.8466
true_beta=7.7488

# Define the inverse logit function
def inv_logit(x):
    return np.exp(x)/(1 + np.exp(x))

# Define the negative log-likelihood function
def neg_log_likelihood(params, x, n, y):
    alpha,beta =params
    theta=inv_logit(alpha+beta*x)
    likelihood=y *np.log(theta)+(n -y)*np.log(1-theta)
    return -np.sum(likelihood)

# Number of simulations
num_simulations=10000

# Data settings from the original dataset
x_values=np.array([-0.86, -0.30, -0.05, 0.73])
n_values=np.array([5, 5, 5, 5])

# Arrays to store estimates
alpha_estimates= np.zeros(num_simulations)
beta_estimates= np.zeros(num_simulations)

for i in range(num_simulations):
    # Simulating y values based on the binomial distribution and the true parameters
    p_values=inv_logit(true_alpha +true_beta *x_values)
    y_simulated=[binom.rvs(n, p) for n, p in zip(n_values,p_values)]
    
    # Estimating parameters for the simulated dataset
    res= minimize(neg_log_likelihood, [0, 0], args=(x_values, n_values, y_simulated), method='L-BFGS-B' )
    alpha_estimates[i], beta_estimates[i]= res.x

# Calculating bias and variance
alpha_bias= np.mean(alpha_estimates)-true_alpha
beta_bias= np.mean(beta_estimates)-true_beta
alpha_variance= np.var(alpha_estimates)
beta_variance= np.var(beta_estimates)

(alpha_bias, beta_bias, alpha_variance, beta_variance)

# Bootstrap Analysis
import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression

# Corrected dataset for demonstration
X = np.array([-0.86, -0.30, -0.05, 0.73]).reshape(-1, 1)  # Doses, reshaped for sklearn compatibility
Y = np.array([0, 1, 3, 5])  # Number of successes  # Binary outcomes, ensuring variability for demonstration

# Function to calculate LD50
def calculate_ld50(alpha, beta):
    return -alpha / beta if beta != 0 else np.nan

# Bootstrap parameters
n_bootstrap_samples = 10000
bootstrap_alpha = np.zeros(n_bootstrap_samples)
bootstrap_beta = np.zeros(n_bootstrap_samples)
bootstrap_ld50 = np.zeros(n_bootstrap_samples)

# Perform corrected bootstrap resampling
for i in range(n_bootstrap_samples):
    X_resampled, Y_resampled = resample(X, Y)
    # Ensure there are at least two classes in the resampled dataset
    if len(np.unique(Y_resampled)) < 2:
        continue
    model = LogisticRegression().fit(X_resampled, Y_resampled)
    alpha = model.intercept_[0]
    beta = model.coef_[0][0]
    ld50 = calculate_ld50(alpha, beta)
    
    bootstrap_alpha[i] = alpha
    bootstrap_beta[i] = beta
    bootstrap_ld50[i] = ld50

# Calculate 95% Confidence Intervals
alpha_ci = np.percentile(bootstrap_alpha, [2.5, 97.5])
beta_ci = np.percentile(bootstrap_beta, [2.5, 97.5])
ld50_ci = np.percentile(bootstrap_ld50, [2.5, 97.5])

(alpha_ci, beta_ci, ld50_ci)

# Logit 
import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom
from sklearn.utils import resample

# Assuming X (doses) and Y (binary outcomes) are defined
X= np.array([-0.86,-0.30,-0.05,0.73])
Y= np.array([0, 1, 3, 5])  

# Define the inverse logit function
def inv_logit(x):
    return np.exp(x)/(1+np.exp(x))

# Define the log-likelihood function for logistic regression
def neg_log_likelihood(params,x,y):
    alpha, beta= params
    p=inv_logit(alpha+beta*x)
    return -np.sum(y*np.log(p)+(1 -y)*np.log(1-p))

# Function to estimate parameters and LD50 for a given dataset
def estimate_parameters(X, Y):
    res= minimize(neg_log_likelihood,[0, 0],args=(X.squeeze(),Y),method='L-BFGS-B')
    alpha,beta= res.x
    ld50= -alpha/beta if beta != 0 else np.nan
    return alpha,beta,ld50

# Bootstrap
n_bootstrap_samples= 10000
bootstrap_results= np.zeros((n_bootstrap_samples, 3)) # For alpha, beta, LD50

for i in range(n_bootstrap_samples):
    X_resampled, Y_resampled= resample(X, Y)
    bootstrap_results[i]= estimate_parameters(X_resampled, Y_resampled)

# Calculate 95% Confidence Intervals
alpha_ci= np.percentile(bootstrap_results[:, 0],[2.5, 97.5])
beta_ci= np.percentile(bootstrap_results[:, 1],[2.5, 97.5])
ld50_ci= np.percentile(bootstrap_results[:, 2] [2.5, 97.5])

print("95% Confidence Intervals:")
print(f"Alpha: {alpha_ci}")
print(f"Beta: {beta_ci}")
print(f"LD50: {ld50_ci}")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the spike and slab components
mu_spike = 0
sigma_spike = 0.1  # Very small variance for the spike
mu_slab = 0
sigma_slab = 2    # Larger variance for the slab

# Generate points for theta
theta = np.linspace(-5, 5, 1000)

# Calculate the density of the spike and slab components
spike_density = norm.pdf(theta, mu_spike, sigma_spike)
slab_density = norm.pdf(theta, mu_slab, sigma_slab)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(theta, spike_density, label='Spike at θ=0', color='red')
plt.plot(theta, slab_density, label='Slab (broader distribution)', color='blue', linestyle='--')
plt.fill_between(theta, spike_density, color='red', alpha=0.3)
plt.fill_between(theta, slab_density, color='blue', alpha=0.1)
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Illustration of a Spike-and-Slab Prior')
plt.legend()
plt.show()

# 11. prior plot
# It seems numpy was not imported, let's correct that and plot again

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the parameters for the prior distribution again
lambda_param= 0.5  # mixture weight
tau1_squared= 0.01  # variance for the spike, making it very narrow
tau2_squared= 5.0  # variance for the slab, making it broader

# Define a range of theta values for plotting
theta_values=np.linspace(-5,5,1000)

# Implement the mixture prior function
def mixture_prior(theta, lambda_param, tau1_squared, tau2_squared):
    spike_component= lambda_param*norm.pdf(theta,0,np.sqrt(tau1_squared))
    slab_component= (1-lambda_param)*norm.pdf(theta,0,np.sqrt(tau2_squared))
    return spike_component+slab_component

# Compute the prior density for each theta value
prior_density=mixture_prior(theta_values,lambda_param,tau1_squared,tau2_squared)

# Plot the prior distribution
plt.figure(figsize=(10,6))
plt.plot(theta_values,prior_density,label='Mixture Prior')
plt.xlabel('Theta')
plt.ylabel('Density')
#plt.title('Mixture Prior Distribution')
plt.legend()
plt.show()


# New parameters for a distinctive explanation
omega = 0.8   # mixture weight
epsilon_squared = 0.02  # New variance for the spike, slightly broader than before
Gamma_squared = 20.0  # New variance for the slab, making it even wider

# New theta range for a comprehensive view
theta_range = np.linspace(-10, 10, 2000)

# Adjusted mixture prior for visualization
def distinctive_mixture_prior(theta, omega, epsilon_squared, Gamma_squared):
    spike = omega * norm.pdf(theta, 0, np.sqrt(epsilon_squared))
    slab = (1 - omega) * norm.pdf(theta, 0, np.sqrt(Gamma_squared))
    return spike + slab

# Calculate the adjusted prior density
distinctive_prior_density = distinctive_mixture_prior(theta_range, omega, epsilon_squared, Gamma_squared)

# Visualization with adjustments
plt.figure(figsize=(12, 7))
plt.plot(theta_range, distinctive_prior_density, 'm-', label='Distinctive Mixture Prior', linewidth=2)
plt.xlabel('$\\theta$', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Distinctive Representation of Mixture Prior Distribution', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

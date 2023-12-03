import pymc3 as pm
import numpy as np

# Define your data
data = np.array([0.49, 0.55, 0.57])

# Define your prior distribution
prior_mean = 0.5
prior_variance = 0.0025

# Define your likelihood function
def likelihood(mu):
    return pm.Normal.dist(mu=mu, sd=np.sqrt(prior_variance)).logp(data).sum()

# Create a PyMC3 model
with pm.Model() as model:
    # Define the prior
    mu = pm.Normal("mu", mu=prior_mean, sd=np.sqrt(prior_variance))
    
    # Define the likelihood
    likelihood_observed = pm.Potential("likelihood_observed", likelihood(mu))
    
    # Sample from the posterior
    trace = pm.sample(2000, tune=1000, cores=1)

# Print summary statistics of the posterior distribution
print(pm.summary(trace))
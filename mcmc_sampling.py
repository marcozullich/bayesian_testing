import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, uniform, norm as normal

if __name__ == "__main__":

    # Define your data
    data = np.array([0.49, 0.55, 0.57])

    # Define your prior distribution
    min_unif_prior = 0.2
    max_unif_prior = 0.8
    prior = uniform(loc=min_unif_prior, scale=max_unif_prior - min_unif_prior)

    # Define your likelihood function
    def likelihood(mu):
        return pm.Normal.dist(mu=mu, sd=data.std()).logp(data).sum()

    # Create a PyMC3 model
    with pm.Model() as model:
        # Define the prior
        mu = pm.Uniform("mu", lower=0.2, upper=0.8)
        
        # Define the likelihood
        likelihood_observed = pm.Potential("likelihood_observed", likelihood(mu))
        
        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, cores=1)

    # Print summary statistics of the posterior distribution
    posterior_samples = trace["mu"]

    kde = gaussian_kde(posterior_samples)


    # Generate x-values for plotting
    # x = np.linspace(posterior_samples.min(), posterior_samples.max(), 100)
    x = np.linspace(0, 1, 1000)

    print(dir(mu))

    posterior_x = kde(x)
    prior_x = prior.pdf(x)
    posterior_nn = normal(loc=0.52978, scale=np.sqrt(0.00047))
    posterior_nn_x = posterior_nn.pdf(x)

    prob_worse_random = (trace["mu"] < 0.5).sum() / len(trace["mu"])

    prob_worse_random_nn = posterior_nn.cdf(0.5)

    print(f"Probability that the true value of mu is worse than random guessing: {prob_worse_random}")
    print(f"Probability that the true value of mu is worse than random guessing for N-N posterior: {prob_worse_random_nn}")

    # Plot the KDE estimate of the posterior distribution
    plt.plot(x, posterior_x, color='blue', label="posterior")
    plt.plot(x, prior_x, color='red', label="prior")
    plt.plot(x, posterior_nn_x, color='orange', label="N-N posterior")
    plt.axvline(x=0.5, color='black', label="mu=0.5", linestyle="--")
    plt.fill_between(x, posterior_x, color='blue', where=(x<0.5), alpha=0.2)
    plt.fill_between(x, posterior_nn_x, color='orange', where=(x<0.5), alpha=0.3)
    plt.xlim(0.4, 0.6)
    plt.xlabel("mu")
    plt.ylabel("Density")
    plt.title("Posterior Distribution")
    plt.legend()
    plt.show()
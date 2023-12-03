# bayesian_testing

Some simple Bayesian testing examples


## Files

### bayesian_posterior_normalnormal.py

Conduct 3 posterior testing on a Normal-Normal set of conjugate prior-likelihood.

The data is a sample of 10 values [0.49, 0.55, 0.57, 0.52, 0.58, 0.56, 0.55, 0.56, 0.57, 0.54].
The likelihood is a Normal with mean `mu` (parameter with prior) and variance `sigma2` (known, hence fixed-- -in this case to the variance of the data). |
`mu` has a Normal prior N(`mu_0`, `sigma2_0`).
The user can choose between different values of the hyperparameters.

#### Usage

| Arg | Description |
| ------------- | ------------- |
| --n_points | Number of datapoints to use. It will limit the data to the first `n_points` in the list above |
| --mu_0 | The mean hyperparameter of the Normal prior (def. 0.5) |
| --sigma2_0 | The variance hyperparameter of the Normal prior (def. 0.0025) |
| --plot_prior | Flag to plot the prior distribution |
| --plot_posterior | Flag to plot the posterior distribution |
| --overlay_plots | Flag to overlay the plots for prior and posterior. If not selected and both --plot_prior and --plot_posterior set, the two will be plot side-by-side |
| --mu_compare | Scalar for the "null hypothesis" of the testing (def. 0.5) |
| --method_compare | Test to execute. Choose between "direct_posterior" (checks probability that model is better than --mu_compare), "credible_interval", "rope" (def. "direct_posterior") |
| --compare_worse | For "direct_posterior", compares whether the model is WORSE than --mu_compare |
| --plot_probability | Plots the posterior & information depending on the chosen testing method |
| --plot_probability_xlim | Restricts x axis on the posterior in the interval --mu_compare ± --plot_probability_xlim |
| --credible_interval_width | Sets the width (in terms of probability) of the credible interval (def. 0.95) |
| --rope_epsilon | Sets the epsilon for the ROPE method. This will define rope as --mu_compare ± --rope_epsilon (def. 0.01) |

Example usage:

`python .\bayesian_posterior_normalnormal.py --plot_prior --plot_posterior --overlay_plots --mu_compare 0.5 --plot_probability --plot_probability_xlim 0.15 --method_compare rope --rope_epsilon 0.02  --n_points 5`

* Prints prior and posterior (overlaying)
* Execute a ROPE testing with epsilon = 0.02

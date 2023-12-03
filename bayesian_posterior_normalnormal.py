import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statistics import mean, variance
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian Posterior")
    
    parser.add_argument('--n_points', type=int, default=10, help='Number of datapoints [1-10] to consider')
    
    parser.add_argument('--mu_0', type=float, default=0.5, help='Prior mean')
    parser.add_argument('--sigma2_0', type=float, default=0.0025, help='Prior variance')
    
    parser.add_argument('--plot_prior', action='store_true', default=False, help='Plot prior')
    parser.add_argument('--plot_posterior', action='store_true', default=False, help='Plot posterior')
    parser.add_argument('--overlay_plots', action='store_true', default=False, help='Overlay plots of prior and posterior')
    
    parser.add_argument('--mu_compare', type=float, default=None, help='Compare posterior mean with this reference value')
    parser.add_argument('--method_compare', choices=["direct_posterior", "credible_interval", "rope"], default="direct_posterior", help='Method to compare posterior mean with reference value')
    
    parser.add_argument('--compare_worse', action='store_true', default=False, help='Compute probability of being worse than mu_compare. If not selected, compute probability of being better than mu_compare')
    parser.add_argument('--plot_probability', action='store_true', default=False, help='Plot probability of posterior mean being greater/smaller than mu_compare')
    parser.add_argument('--plot_probability_xlim', default=None, help='Set x-axis +- quantity around mu_compare')
    
    parser.add_argument('--credible_interval_width', default=0.95, type=float, help='The credibility interval width for the credible interval method')
    
    parser.add_argument('--rope_epsilon', default=0.01, type=float, help='The width of the ROPE (Region Of Practical Equivalence) for the ROPE method')
    args = parser.parse_args()
    return args

def get_posterior_params(mu, sigma2, num_data, mu_0, sigma2_0):
    '''
    Given the prior parameters (mu_0, sigma20) and the evidence (mu, sigma2, num_data), compute the posterior parameters mu_post, and sigma2_post.
    '''
    sigma2_post = 1/(1/sigma2_0 + num_data/sigma2)
    mu_post = sigma2_post*(mu_0/sigma2_0 + num_data*mu/sigma2)
    return mu_post, sigma2_post

def plot_distribution(distribution, x, label, color, ax=None):
    '''
    Returns a plot object for the distribution
    '''
    if ax is None:
        ax = plt.gca()
    line,  = ax.plot(x, distribution.pdf(x), label=label, color=color)
    return line

def posterior_mean_probability(mu_compare, posterior_distribution, compare_worse):
    '''
    Returns the probability that mu_compare is greater/smaller than mu_post.
    '''
    prob_level = posterior_distribution.cdf(mu_compare)
    if compare_worse:
        return prob_level
    else:
        return 1-prob_level
    
def interval_intersection(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    
    if start <= end:
        return (start, end)
    else:
        return None
    

def plot_distributions(prior, posterior, args):
    x = np.linspace(0, 1, 1000)
    if args.plot_prior or args.plot_posterior:
        if args.overlay_plots or (not args.plot_posterior or not args.plot_prior):
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        if args.plot_prior:
            axis = ax if args.overlay_plots or (not args.plot_posterior) else ax1
            plot_distribution(prior, x, 'Prior', 'blue', axis)
            axis.legend()
        if args.plot_posterior:
            axis = ax if args.overlay_plots or (not args.plot_prior) else ax2
            plot_distribution(posterior, x, 'Posterior', 'red', axis)
            axis.legend()
        
        plt.show()
        
def execute_posterior_test(posterior, args):
    x = np.linspace(0, 1, 1000)
    if args.mu_compare is not None:
        if args.plot_probability:
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
            plot_distribution(posterior, x, 'Posterior', 'red', ax)
            
        if args.method_compare == "direct_posterior":
            prob_mu_compare = posterior_mean_probability(args.mu_compare, posterior, args.compare_worse)
            comparison_goal = "worse" if args.compare_worse else "better"
            print(f"Plausibility of model being {comparison_goal} than {args.mu_compare}: {prob_mu_compare}")
            
            if args.plot_probability:
                plt.axvline(x=args.mu_compare, color='black', linestyle='--')
                if args.compare_worse:
                    ax.fill_between(x, 0, posterior.pdf(x), where=(x < args.mu_compare), color='red', alpha=0.2)
                else:
                    ax.fill_between(x, 0, posterior.pdf(x), where=(x > args.mu_compare), color='red', alpha=0.2)
                if args.plot_probability_xlim is not None:
                    ax.set_xlim([args.mu_compare - float(args.plot_probability_xlim), args.mu_compare + float(args.plot_probability_xlim)])
                plt.show()
        else:
            ci_limits = posterior.ppf((1-args.credible_interval_width)/2), posterior.ppf(1-(1-args.credible_interval_width)/2)
            if args.method_compare == "credible_interval":
                mu_compare_in = ci_limits[0] < args.mu_compare < ci_limits[1]
                mu_compare_status = "in" if mu_compare_in else "out of"
                print(f"Credible interval: [{ci_limits[0]} ; {ci_limits[1]}]. {args.mu_compare} is {mu_compare_status} the credible interval.")
                
                
                if args.plot_probability:
                    plt.axvline(x=ci_limits[0], color='black', linestyle='--', label=f"{args.credible_interval_width*100}% CI")
                    plt.axvline(x=ci_limits[1], color='black', linestyle='--')
                    plt.axvline(x=args.mu_compare, color='blue', linestyle=':', label=f"hypothesis mu = {args.mu_compare}")
                    if args.plot_probability_xlim is not None:
                        ax.set_xlim([args.mu_compare - float(args.plot_probability_xlim), args.mu_compare + float(args.plot_probability_xlim)])
                    plt.legend()
                    plt.show()

            elif args.method_compare == "rope":
                rope_limits = args.mu_compare - args.rope_epsilon, args.mu_compare + args.rope_epsilon
                intersection = interval_intersection(ci_limits, (rope_limits[0], rope_limits[1]))
                if intersection is None:
                    print(f"CI and ROPE do not intersect. ROPE: [{rope_limits[0]} ; {rope_limits[1]}]. CI: [{ci_limits[0]} ; {ci_limits[1]}]. Reject hypothesis that mu = {args.mu_compare}.")
                else:
                    if (intersection[0] == ci_limits[0] and intersection[1] == ci_limits[1]) or (intersection[0] == rope_limits[0] and intersection[1] == rope_limits[1]):
                        print(f"Intersection coincides with ROPE or CI. ROPE: [{rope_limits[0]} ; {rope_limits[1]}]. CI: [{ci_limits[0]} ; {ci_limits[1]}]. Intersection: [{intersection[0]} ; {intersection[1]}]. Accept hypothesis that mu = {args.mu_compare}.")
                    else:
                        print(f"Intersection exists. ROPE: [{rope_limits[0]} ; {rope_limits[1]}]. CI: [{ci_limits[0]} ; {ci_limits[1]}]. Intersection: [{intersection[0]} ; {intersection[1]}]. Withold judgement on hypothesis that mu = {args.mu_compare}.")
                
                if args.plot_probability:
                    plt.axvline(x=ci_limits[0], color='black', linestyle='--', label=f"{args.credible_interval_width*100}% CI")
                    plt.axvline(x=ci_limits[1], color='black', linestyle='--')
                    plt.axvline(x=rope_limits[0], color='blue', linestyle=':', label=f"ROPE {args.mu_compare} ± {args.rope_epsilon}")
                    plt.axvline(x=rope_limits[1], color='blue', linestyle=':')
                    if args.plot_probability_xlim is not None:
                        ax.set_xlim([args.mu_compare - float(args.plot_probability_xlim), args.mu_compare + float(args.plot_probability_xlim)])
                    plt.legend()
                    plt.show()

if __name__ == "__main__":
    args = parse_args()
    
    if args.n_points < 1 or args.n_points > 10:
        raise ValueError(f"Number of datapoints must be in [1, 10]. Found: {args.n_points}")
    data = [0.49, 0.55, 0.57, 0.52, 0.58, 0.56, 0.55, 0.56, 0.57, 0.54][:args.n_points]
    mu = mean(data)
    sigma2 = variance(data)
    
    print(f"Data mean: {mu}, Data variance: {sigma2} [num data points: {len(data)}]")
    
    prior = stats.norm(loc=args.mu_0, scale=np.sqrt(args.sigma2_0))
    
    mu_post, sigma2_post = get_posterior_params(mu, sigma2, len(data), args.mu_0, args.sigma2_0)
    posterior = stats.norm(loc=mu_post, scale=np.sqrt(sigma2_post))
    print(f"Posterior mean: {mu_post}, Posterior variance: {sigma2_post}")
    
    plot_distributions(prior, posterior, args)
    
    execute_posterior_test(posterior, args)
    

    
                        
            
    
    
    
    
    
    
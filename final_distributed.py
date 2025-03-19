import numpy as np
import pymc3 as pm
import time
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Function to generate synthetic data
def generate_data(n, d):
    X = np.random.randn(n, d)
    theta_true = np.random.randn(d)
    noise = np.random.normal(0, 1.0, size=n)
    y = X @ theta_true + noise
    return X, y, theta_true

# Function to add Laplace noise for differential privacy
def add_laplace_noise(y, epsilon=1.0, sensitivity=1.0):
    b = sensitivity / epsilon
    noise = np.random.laplace(0, b, size=len(y))
    return y + noise

# Function: Pseudo-Marginal MCMC
def pseudo_marginal_mcmc(X, y, epsilon=1.0, samples=3000):
    d = X.shape[1]
    y_noisy = add_laplace_noise(y, epsilon=epsilon)

    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0, sigma=5, shape=d)
        y_obs = pm.Normal("y_obs", mu=pm.math.dot(X, theta), sigma=1.0 + 1e-3, observed=y_noisy)
        start = {"theta": np.random.randn(d) * 0.01}
        start_time = time.time()
        trace = pm.sample(samples, step=pm.Metropolis(), tune=2000, cores=1,
                           progressbar=False, return_inferencedata=False)
        runtime = time.time() - start_time

    try:
        ess = az.ess(trace).to_array().mean().item()
        rhat = az.rhat(trace).to_array().mean().item()
    except:
        ess, rhat = np.nan, np.nan

    return runtime, ess, rhat

# Function: Metropolis-Hastings
def metropolis_hastings(X, y, samples=3000):
    d = X.shape[1]

    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0, sigma=5, shape=d)
        y_obs = pm.Normal("y_obs", mu=pm.math.dot(X, theta), sigma=1.0, observed=y)

        start_time = time.time()
        trace = pm.sample(samples, step=pm.Metropolis(), tune=2000, cores=1,
                          progressbar=False, return_inferencedata=False)
        runtime = time.time() - start_time

    try:
        ess = az.ess(trace).to_array().mean().item()
        rhat = az.rhat(trace).to_array().mean().item()
    except:
        ess, rhat = np.nan, np.nan

    return runtime, ess, rhat

# Function: HMC/NUTS
def hmc_nuts(X, y, samples=2000):
    d = X.shape[1]

    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0, sigma=5, shape=d)
        y_obs = pm.Normal("y_obs", mu=pm.math.dot(X, theta), sigma=1.0, observed=y)

        start_time = time.time()
        trace = pm.sample(samples, step=pm.NUTS(), tune=1000, cores=1,
                          progressbar=False, return_inferencedata=False)
        runtime = time.time() - start_time

    try:
        ess = az.ess(trace).to_array().mean().item()
        rhat = az.rhat(trace).to_array().mean().item()
    except:
        ess, rhat = np.nan, np.nan

    return runtime, ess, rhat

# ------------------ DISTRIBUTED COMPUTING SETUP ------------------

# Simulate distributed partitions (3 hospitals)
def split_data(X_all, y_all):
    X_A, y_A = X_all[:500], y_all[:500]
    X_B, y_B = X_all[500:1000], y_all[500:1000]
    X_C, y_C = X_all[1000:], y_all[1000:]
    return (X_A, y_A), (X_B, y_B), (X_C, y_C)

n_total = 1500  # total rows
d_values = [10, 50, 100, 200]
epsilons = [0.1, 1, 10]
distributed_results = []

# Run models for each hospital and dimension
for d in d_values:
    print(f"\nRunning Distributed MCMC for d = {d}")
    X_all, y_all, _ = generate_data(n_total, d)
    partitions = split_data(X_all, y_all)
    hospitals = ['Hospital A', 'Hospital B', 'Hospital C']

    for i, (X_part, y_part) in enumerate(partitions):
        print(f"  Running for {hospitals[i]}")

        # Metropolis-Hastings
        mh_time, mh_ess, mh_rhat = metropolis_hastings(X_part, y_part)
        distributed_results.append([hospitals[i], d, 'MH', 'N/A', mh_time, mh_ess, mh_rhat])

        # HMC/NUTS
        hmc_time, hmc_ess, hmc_rhat = hmc_nuts(X_part, y_part)
        distributed_results.append([hospitals[i], d, 'HMC/NUTS', 'N/A', hmc_time, hmc_ess, hmc_rhat])

        # PM-MCMC with varying ε
        for eps in epsilons:
            pm_time, pm_ess, pm_rhat = pseudo_marginal_mcmc(X_part, y_part, epsilon=eps)
            distributed_results.append([hospitals[i], d, 'PM-MCMC', eps, pm_time, pm_ess, pm_rhat])

# ------------------- FINAL RESULTS TABLE -------------------
df_dist = pd.DataFrame(distributed_results, columns=["Hospital", "Dimensionality", "Method", "Epsilon", "Time (s)", "ESS", "R-hat"])

print("\nDistributed MCMC Summary:")
print(df_dist)

# Optionally save to CSV
df_dist.to_csv("distributed_mcmc_results.csv", index=False)

# ------------------- Optional Plot -------------------
# You can later add custom plots using df_dist for hospital-wise visualization.

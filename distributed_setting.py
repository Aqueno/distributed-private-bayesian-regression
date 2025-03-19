import numpy as np
import pymc3 as pm
import time
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


# Generate synthetic linear regression data
def generate_data(n, d):
    X = np.random.randn(n, d)
    theta_true = np.random.randn(d)
    noise = np.random.normal(0, 1.0, size=n)
    y = X @ theta_true + noise
    return X, y, theta_true


# Add Laplace noise for Differential Privacy
def add_laplace_noise(y, epsilon=1.0, sensitivity=1.0):
    b = sensitivity / epsilon
    noise = np.random.laplace(0, b, size=len(y))
    return y + noise


# Pseudo-Marginal MCMC on one partition
def pseudo_marginal_mcmc(X, y, epsilon=1.0, samples=2000):
    d = X.shape[1]
    y_noisy = add_laplace_noise(y, epsilon=epsilon)

    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0, sigma=5, shape=d)
        y_obs = pm.Normal("y_obs", mu=pm.math.dot(X, theta), sigma=1.0 + 1e-3, observed=y_noisy)

        start = {"theta": np.random.randn(d) * 0.01}
        start_time = time.time()
        trace = pm.sample(samples, step=pm.Metropolis(), tune=1000, cores=1,
                          initvals=start, progressbar=False, return_inferencedata=True)
        runtime = time.time() - start_time

    try:
        ess = az.ess(trace).to_array().mean().item()
        rhat = az.rhat(trace).to_array().mean().item()
    except:
        ess, rhat = np.nan, np.nan

    return trace, runtime, ess, rhat


# Simulate distributed data partitions (e.g., 3 hospitals)
n_total, d = 1500, 50
X_all, y_all, theta_true = generate_data(n_total, d)

X_A, y_A = X_all[:500], y_all[:500]
X_B, y_B = X_all[500:1000], y_all[500:1000]
X_C, y_C = X_all[1000:], y_all[1000:]

# Run PM-MCMC independently on each partition
epsilons = [0.1, 1, 10]
results = []

for eps in epsilons:
    print(f"Running PM-MCMC on Hospital A with ε={eps}")
    trace_A, time_A, ess_A, rhat_A = pseudo_marginal_mcmc(X_A, y_A, epsilon=eps)

    print(f"Running PM-MCMC on Hospital B with ε={eps}")
    trace_B, time_B, ess_B, rhat_B = pseudo_marginal_mcmc(X_B, y_B, epsilon=eps)

    print(f"Running PM-MCMC on Hospital C with ε={eps}")
    trace_C, time_C, ess_C, rhat_C = pseudo_marginal_mcmc(X_C, y_C, epsilon=eps)

    results.extend([
        ("Hospital A", eps, time_A, ess_A, rhat_A),
        ("Hospital B", eps, time_B, ess_B, rhat_B),
        ("Hospital C", eps, time_C, ess_C, rhat_C),
    ])

# Display results
df = pd.DataFrame(results, columns=["Partition", "Epsilon", "Time (s)", "ESS", "R-hat"])
print("\nDistributed PM-MCMC Results:")
print(df)

# Optional: plot runtime vs epsilon
for hospital in df["Partition"].unique():
    subset = df[df["Partition"] == hospital]
    plt.plot(subset["Epsilon"], subset["Time (s)"], marker='o', label=hospital)

plt.xlabel("Privacy Parameter (ε)")
plt.ylabel("Runtime (s)")
plt.title("Runtime vs Privacy across Distributed Nodes")
plt.legend()
plt.grid()
plt.show()

Distributed Private Bayesian Regression

This repository contains the source code for **Distributed Differentially Private Bayesian Regression using Advanced MCMC Techniques**.

Overview

The code implements and compares the performance of three MCMC methods:
- Metropolis-Hastings (MH)
- Hamiltonian Monte Carlo (HMC/NUTS)
- Pseudo-Marginal MCMC (PM-MCMC)

across distributed nodes (e.g., Hospital A, B, C) with varying feature dimensionalities and differential privacy budgets.

Purpose

The goal is to simulate a privacy-preserving distributed Bayesian inference setting.  
Sampling performance is evaluated based on:
- Runtime
- Effective Sample Size (ESS)
- R-hat (convergence diagnostic)

Some hyperparameters and sample sizes were intentionally tuned to observe runtime behavior, particularly for MH under high-dimensional settings.
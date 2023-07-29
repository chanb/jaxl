# Running Preliminary Experiments

## PPO Objective Comparison
We compare the following PPO objective in discrete action space:
1. Clipping objective
1. Reverse-KL objective

## Necessity of Finetuning in Pendulum
We compare the performance change in transferring an expert policy from a source environment to target environments

## Behavioural Cloning (BC) Ablation
### The amount of demonstration data required
How much data do we need in order to achieve near-expert policy?

### The impact of subsampling scheme
How does subsampling scheme impact the sample efficiency of BC?

# Structure
We assume the working directory is the directory containing this README:
```
.
├── bc_amount_data/
├── bc_subsampling
├── expert_policies/
├── objective_comparison/
├── policy_robustness_pendulum/
└── README.md
```

# Experiment Log
## BC vs MTBC
We hold out 10 environment variants for each experiment.
- Held-out environments: $\{0, \dots, 9\}$
- Training environments: $\{10, \dots, 99\}$

In general, we perform experiments such that the total amount of data matches the BC data ablation study (i.e. the empirical estimate of the minimum samples required to successfully imitate expert policies.)

## Number of Training Tasks
- Fix a specific

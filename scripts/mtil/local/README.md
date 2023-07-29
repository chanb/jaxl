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

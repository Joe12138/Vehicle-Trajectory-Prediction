# <center>The mathematical model of vehicle trajectory prediction</center>

## Problem Formulation

## Baseline

$\mathcal{A}$: on-road agents

$\mathcal{S}$: observed states of on-road agents $\mathcal{A}$

$\mathcal{M}$: HD map

$\mathbf{s}_i^t$: denotes the state $a_i \in \mathcal{A}$ at frame $t$, including position, heading, velocity, turning rate and actor type

$\mathbf{s}_i=\{s_i^{-T_P+1}, s_i^{-T_P+2},...,s_i^0\}$: denotes the state sequence in the observed period $T_P$

#### Question

Given any agent as the prediction target, we denote it by $a_{tar}$ and its surrounding agents by $\mathcal{A_{nbrs}}=\{a_1, a_2,...,a_m\}$ for differentiation, with their state sequence correspondingly given as $s_{tar}$ and $s_{nbrs}=\{s_1, s_2,...,s_m\}$.

$\mathcal{S}=\{s_{tar}\} \cup \mathcal{S}_{nbrs}$ and $\mathcal{A}=\{a_{tar}\} \cup \mathcal{A}_{nbrs}$

#### Objective

To predict multi-modal future trajectories $\mathcal{T}_{tar}=\{\mathcal{T}_k|k=1,2,...,K\}$ together with corresponding trajectory probability $\{p_k\}$, where $\mathcal{T}_k$ denotes a predicted trajectory for target agent $a_{tar}$ with continuous state information up to the prediction horizon $T_{F}$

$K$ is the number of predicted trajectories.

Additionally, it is required to ensure each prediction $\mathcal{T}_{k}\in \mathcal{T}_{tar}$ is feasible with existing constraints $\mathcal{C}$, which includes environment constraints $\mathcal{C}_{\mathcal{M}}$ and the kinematic constraints $\mathcal{C}_{tar}$.


# Towards Learning-Based Energy-Efficient Online Coordinated Virtual Network Embedding Framework

## Overview:
This project presents a Reinforcement Learning (RL)-based framework for solving the Virtual Network Embedding (VNE) problem efficiently and dynamically. 
It implements PPO using Stable-Baselines3 to achieve:

1. Higher acceptance ratio
2. Better revenue-to-cost performance
3. Lower energy consumption

The goal is to prove that our learning-based VNE algorithm is the optimal approach for online, energy-aware embedding — validated through simulation and comparative performance graphs.

## Motivation:
Traditional heuristic-based VNE algorithms are:
- Static and rule-dependent
- Poorly scalable to dynamic environments
- Energy-inefficient

This project uses Deep Reinforcement Learning to learn adaptive strategies that efficiently embed virtual networks on dynamic substrate networks while reducing overall power usage.

## Key Features:
- Custom Gym-based environment for VNE
- Stable-Baselines3 PPO agents
- Dynamic Virtual Network Requests (VNRs)
  
- Comprehensive evaluation metrics:
  * Acceptance Ratio
  * Revenue
  * Revenue-to-Cost Ratio
  * Energy Consumption
  * Revenue-to-Energy (R2E) Coefficient

## Architecture Overview:
Input: Virtual Network Requests (VNRs)
```
Custom Gym Environment (VNEEnv)
↓
RL Agent (PPO )
↓
Action: Node & Link Mapping
↓
Reward: Energy + Performance Optimization
↓
Training & Policy Improvement
↓
Evaluation & Visualization
```
## Tools and Libraries Used:
Core: random, heapq, os, warnings

Numerical: numpy, scipy

Network Simulation: networkx

Reinforcement Learning: gym, stable-baselines3

Visualization: matplotlib

Backend: torch (PyTorch)

## To install all dependencies:
```bash
pip install numpy scipy networkx gym matplotlib stable-baselines3 torch
```

## Implementation Steps:
1. Substrate Network
   - Built using NetworkX with nodes and links assigned CPU, bandwidth, and energy cost parameters.
2. Virtual Network Requests (VNRs)
   - Randomly generated topologies with configurable node/link counts and resource demands.
   - Requests arrive and expire dynamically.
3. Reinforcement Learning Environment
   - Custom gym.Env defining observation and action spaces.
   - Reward balances resource efficiency, energy usage, and VNR acceptance.
4. Algorithms Used
   - PPO (Proximal Policy Optimization)
   - A2C (Advantage Actor-Critic) – baseline agent
   - DRL_VNE
   - DRL_SFCP – energy-scaled PPO variant
5. Evaluation
   - Tested on large (100-node) topologies with visualized metrics:
     * Acceptance Ratio vs. VNRs
     * Revenue vs. VNRs
     * Revenue-to-Cost Ratio
     * Energy Consumption
     * R2E Coefficient

The PPO-based energy-aware model achieved:
- Highest acceptance ratio
- Best revenue efficiency
- Significant energy savings

## Key Takeaways:
- RL agents learn optimal embedding policies dynamically.
- PPO-based VNE model delivers superior energy efficiency.
- Framework scales smoothly to large network topologies.
- Demonstrates optimal and generalizable embedding performance.

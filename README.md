# Reinforcement Learning for Physics Simulations: A Comparison of A2C, PPO, and TRPO

This project presents a comparative analysis of three popular reinforcement learning (RL) algorithms: **Advantage Actor-Critic (A2C)**, **Proximal Policy Optimization (PPO)**, and **Trust Region Policy Optimization (TRPO)** in the Acrobot environment. The main goal is to evaluate and compare the performance and efficiency of these algorithms by testing them in a challenging physics simulation environment. This project aims to understand the strengths and limitations of A2C, PPO, and TRPO algorithms by testing them in the Acrobot environment, a common benchmark in RL research. The project also highlights the impact of hyperparameter tuning on the performance of these algorithms.

## Algorithms Compared

1. **A2C (Advantage Actor-Critic)**: Combines policy and value-based methods to optimize the learning process. Known for low variance and fast convergence.
   
2. **PPO (Proximal Policy Optimization)**: An on-policy algorithm known for stability and sample efficiency. It adjusts policies iteratively to enhance performance.

3. **TRPO (Trust Region Policy Optimization)**: Uses a trust region constraint to maintain conservative policy updates, enhancing stability and preventing policy collapse.

## Environment

- **Acrobot**: A two-link arm simulation where the agent must swing the arm up past a certain point. This environment is complex due to its high-dimensional state space and non-linear dynamics.

## Libraries

- **Torch**: 2.1.0.dev20230307+cu117
- **TensorBoard**: 2.12.0
- **Stable-Baselines3**: 2.0.0a5
- **sb3-contrib**: 2.0.0a4
- **PyGame**: 2.1.3.dev8
- **Gym**: 0.26.2

#!/usr/bin/env python3
import numpy as np

def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Parameters:
    - env: OpenAI Gym environment
    - V: numpy.ndarray of shape (s,) containing value estimates
    - policy: function(state) -> action
    - episodes: number of training episodes
    - max_steps: max steps per episode
    - alpha: learning rate
    - gamma: discount factor

    Returns:
    - Updated value function V
    """

    for _ in range(episodes):
        state = env.reset()
        episode = []

        # Generate one episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            episode.append((state, reward))
            state = next_state

            if done:
                break

        # Compute returns and update (First-Visit MC)
        G = 0
        visited_states = set()

        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = gamma * G + r

            if s not in visited_states:
                visited_states.add(s)
                V[s] += alpha * (G - V[s])

    return V

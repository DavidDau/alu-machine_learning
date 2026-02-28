#!/usr/bin/env python3
import numpy as np

def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm.

    Parameters:
    - env: OpenAI Gym environment
    - V: numpy.ndarray of shape (s,) containing value estimates
    - policy: function(state) -> action
    - lambtha: eligibility trace factor
    - episodes: number of episodes
    - max_steps: max steps per episode
    - alpha: learning rate
    - gamma: discount factor

    Returns:
    - Updated value function V
    """

    n_states = V.shape[0]

    for _ in range(episodes):
        state = env.reset()
        E = np.zeros(n_states)  # eligibility traces

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            # TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Accumulate trace
            E[state] += 1

            # Update all states
            V += alpha * delta * E

            # Decay traces
            E *= gamma * lambtha

            state = next_state

            if done:
                break

    return V

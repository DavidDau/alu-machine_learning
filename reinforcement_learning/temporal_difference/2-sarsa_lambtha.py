#!/usr/bin/env python3
import numpy as np

def sarsa_lambtha(env, Q, lambtha,
                  episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(λ)

    Parameters:
    - env: OpenAI Gym environment
    - Q: numpy.ndarray of shape (s, a)
    - lambtha: eligibility trace factor
    - episodes: number of episodes
    - max_steps: max steps per episode
    - alpha: learning rate
    - gamma: discount factor
    - epsilon: initial epsilon for ε-greedy
    - min_epsilon: minimum epsilon
    - epsilon_decay: decay rate per episode

    Returns:
    - Updated Q table
    """

    n_states, n_actions = Q.shape

    def epsilon_greedy(state, eps):
        if np.random.uniform(0, 1) < eps:
            return np.random.randint(n_actions)
        return np.argmax(Q[state])

    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(state, epsilon)

        # Eligibility traces
        E = np.zeros((n_states, n_actions))

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(next_state, epsilon)

            # TD error (on-policy)
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Accumulate trace
            E[state, action] += 1

            # Update all Q values
            Q += alpha * delta * E

            # Decay eligibility traces
            E *= gamma * lambtha

            state = next_state
            action = next_action

            if done:
                break

        # Decay epsilon per episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
    
from typing import List, Tuple

def compute_gae(
    rewards: List[float],
    dones: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE) and returns.
    
    Args:
        rewards (List[float]): List of rewards collected during rollout, length T.
        dones (List[float]): List of masks (T), where 1 indicates non-terminal (not done).
        values (List[float]): List of predicted state values for each state (T).
        gamma (float, optional): Discount factor for future rewards. Default is 0.99.
        lam (float, optional): GAE lambda parameter controlling bias-variance trade-off. Default is 0.95.
        
    Returns:
        Tuple[List[float], List[float]]:
            - returns (List[float]): Discounted sum of rewards (target for value function).
            - advantages (List[float]): Estimated advantages used to update the policy.
    """
    T = len(rewards)
    # Initialize the advantages list with zeros
    advantages = [0.0] * T
    gae = 0.0  # Initialize GAE accumulator

    # Iterate over the rollout in reverse to compute GAE
    for t in reversed(range(T)):
        # Temporal Difference (TD) error
        delta = rewards[t] + gamma * values[t+1] * dones[t] - values[t]
        # Update GAE using the TD error and previous GAE value
        gae = delta + gamma * lam * dones[t] * gae
        advantages[t] = gae

    # Compute the returns by adding advantages to the value estimates
    returns = [advantages[t] + values[t] for t in range(T)]
    return returns, advantages


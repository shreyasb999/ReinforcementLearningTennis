# Rollout buffer to store experiences collected during training
# Used for PPO updates

import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        """
        Initializes the buffer by clearing all stored experiences.
        """
        self.clear()

    def clear(self):
        """
        Clears all stored experiences from the buffer.
        """
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.dones     = []
        self.values    = []

    def add(self, state, action, log_prob, reward, done, value, device):
        """
        Adds a single experience to the buffer.
        
        Args:
            state (np.ndarray): Environment state
            action (np.ndarray or Tensor): Action taken
            log_prob (float or Tensor): Log probability of the action
            reward (float): Reward received
            done (bool): Done flag (True if episode ended)
            value (float): Estimated state value
            device (torch.device): Device to store tensors (CPU or GPU)
        """
        # Convert state to tensor and ensure it is 1-D
        st = torch.as_tensor(state, dtype=torch.float32, device=device)
        st = st.view(-1)    # shape: [state_dim]
        self.states.append(st)

        # Convert action to tensor if it is a numpy array and ensure it is 1-D
        if isinstance(action, np.ndarray):
            ac = torch.as_tensor(action, dtype=torch.float32, device=device).view(-1)
        else:
            ac = action.view(-1)
        self.actions.append(ac)

        # Ensure log probability is a scalar tensor
        if isinstance(log_prob, torch.Tensor):
            lp = log_prob.item()
        else:
            lp = float(log_prob)
        self.log_probs.append(torch.tensor(lp, dtype=torch.float32, device=device))

        # Convert reward, done, and value to tensors
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        # Store 1.0 - done to make masks (1 if not done, 0 if done)
        self.dones.append(torch.tensor(1.0 - done, dtype=torch.float32, device=device))
        self.values.append(torch.tensor(value, dtype=torch.float32, device=device))

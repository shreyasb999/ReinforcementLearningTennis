import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .buffer import RolloutBuffer
from .utils import compute_gae

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent for continuous action spaces.
    This class implements a shared feature extractor, separate actor and critic heads,
    and uses Generalized Advantage Estimation (GAE) for computing policy updates.
    """
    def __init__(self, state_size, action_size, config):
        # Select device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Hyperparameters from config ---
        hidden_sizes = config["network"]["hidden_sizes"]           # Sizes for shared MLP layers
        lr     = float(config["ppo"]["lr"])                        # Learning rate for actor/critic
        self.gamma   = float(config["ppo"]["gamma"])               # Discount factor
        self.lam     = float(config["ppo"]["gae_lambda"])          # GAE lambda
        self.ent_coef= float(config["ppo"]["initial_entropy_coef"]) # Entropy coefficient

        # --- Shared trunk network ---
        # Maps state to a common feature representation used by both actor & critic
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        ).to(self.device)

        # --- Actor head ---
        # Predicts mean of Gaussian policy
        self.actor = nn.Linear(hidden_sizes[1], action_size).to(self.device)

        # --- Critic head ---
        # Predicts state-value estimate
        self.critic = nn.Linear(hidden_sizes[1], 1).to(self.device)

        # --- Action log‐standard deviation parameter ---
        # Learned parameter controlling exploration noise
        self.log_std = nn.Parameter(torch.zeros(1, action_size, device=self.device))

        # --- Optimizer ---
        # Jointly optimizes shared trunk, actor, critic, and log_std
        self.optimizer = torch.optim.Adam(
            list(self.shared.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()) +
            [self.log_std],
            lr=lr
        )

        # --- Rollout buffer ---
        # Stores per-step trajectories for one update
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        """
        Selects an action given the current state by sampling from a Gaussian policy.
        Also returns the action's log-probability and the critic's value estimate.
        """
        # Convert state to tensor and add batch dimension
        s = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # [1, state_dim]

        with torch.no_grad():
            # Shared feature extraction
            feat = self.shared(s)  
            # Actor: compute mean and sample action
            mu   = self.actor(feat)                 # [1, action_dim]
            std  = torch.exp(self.log_std)          # [1, action_dim]
            dist = Normal(mu, std)
            act_t= dist.sample()                    # [1, action_dim]
            lp_t = dist.log_prob(act_t).sum(dim=-1) # [1] sum over action_dim
            # Critic: compute state-value
            v    = self.critic(feat).squeeze(-1)    # [1]

        # Convert to numpy / Python scalars
        action = act_t.squeeze(0).cpu().numpy()    # [action_dim]
        logp   = lp_t.item()                       # float
        value  = v.item()                          # float
        return action, logp, value

    def update(self):
        """
        Performs a single A2C update:
         1) Compute GAE advantages and discounted returns
         2) Compute policy and value losses
         3) Take an optimizer step
         4) Clear the rollout buffer
        """
        # 1) Stack rollout tensors
        states   = torch.stack(self.buffer.states).to(self.device)    # [T, sdim]
        actions  = torch.stack(self.buffer.actions).to(self.device)   # [T, adim]

        # 2) Bootstrap last value for GAE
        with torch.no_grad():
            last_s    = self.buffer.states[-1].unsqueeze(0).to(self.device)
            next_feat = self.shared(last_s)
            last_val  = self.critic(next_feat).squeeze(-1).item()

        # 3) Compute returns & advantages via GAE
        returns, advs = compute_gae(
            rewards=self.buffer.rewards,
            dones=self.buffer.dones,
            values=self.buffer.values + [last_val],
            gamma=self.gamma,
            lam=self.lam
        )
        returns = torch.FloatTensor(returns).to(self.device)  # [T]
        advs    = torch.FloatTensor(advs).to(self.device)     # [T]
        advs    = (advs - advs.mean()) / (advs.std() + 1e-8)  # normalize advantages

        # 4) Single-step update (no PPO clipping or multiple epochs)
        feat    = self.shared(states)       # [T, hidden]
        mu       = self.actor(feat)         # [T, action_dim]
        std      = torch.exp(self.log_std).expand_as(mu)
        dist     = Normal(mu, std)
        logp     = dist.log_prob(actions).sum(dim=-1)  # [T]
        entropy  = dist.entropy().sum(dim=-1).mean()   # scalar
        vals     = self.critic(feat).squeeze(-1)       # [T]

        # 5) Compute losses
        actor_loss  = -(logp * advs).mean()            # policy gradient loss
        critic_loss = F.mse_loss(vals, returns)        # value function loss (MSE)
        loss        = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy

        # 6) Backpropagation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 7) Clear buffer for next rollout
        self.buffer.clear()

    def save(self, filepath: str):
        """
        Save the shared trunk, actor, critic, log_std, and optimizer state to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'shared_state_dict':    self.shared.state_dict(),
            'actor_state_dict':     self.actor.state_dict(),
            'critic_state_dict':    self.critic.state_dict(),
            'log_std':              self.log_std.detach().cpu(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"[A2CAgent] Model saved to {filepath}")

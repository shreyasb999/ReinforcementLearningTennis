import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from models.actor_critic import ActorCritic
from ppo.buffer import RolloutBuffer
import numpy as np
import csv
from typing import Optional
from ppo.utils import compute_gae
import datetime




class PPOAgent:
    def __init__(self, state_size: int, action_size: int, config: dict):
        """
        Initializes the PPO agent, including actor-critic networks, optimizer, and buffer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- pull hyperparameters from config ---
        hidden_dim    = int(config["network"].get("hidden_size", 256))
        lr            = float(config["ppo"]["lr_actor"])
        self.gamma    = float(config["ppo"]["gamma"])
        self.lam      = float(config["ppo"]["gae_lambda"])
        self.eps_clip = float(config["ppo"]["clip_eps"])
        self.K_epochs = int(config["ppo"]["epochs"])
        self.base_ent_coef = float(config["ppo"]["initial_entropy_coef"])

        # --- define separate actor and critic networks ---
        self.actor = ActorCritic(state_size, hidden_dim, action_size,
                                 output_gate=torch.tanh).to(self.device)
        self.critic = ActorCritic(state_size, hidden_dim, 1,
                                  output_gate=None).to(self.device)

        # log standard deviation for Gaussian policy
        self.log_std = nn.Parameter(
            torch.ones(1, action_size, device=self.device) * -1.0
        )

        # single optimizer for actor, critic, and log_std
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters()) +
            [self.log_std],
            lr=lr
        )

        # rollout buffer to store trajectories
        self.buffer = RolloutBuffer()

        # setup CSV logging for training statistics
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M")
        csv_dir = config["train"].get("ppo_logs_dir", "logs/agents/")
        filename = f"ppo_agent_{id(self)}_{date_str}.csv"
        os.makedirs(csv_dir, exist_ok=True)
        self.csv_path = os.path.join(csv_dir, filename)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "episode", "epoch",
                "actor_loss", "critic_loss",
                "entropy", "total_loss", "kl"
            ])

    def select_action(self, state):
        """
        Selects an action given a state using the actor network.
        
        Returns:
            action (np.ndarray): Selected action
            log_prob (float): Log probability of the action
            value (float): State value estimate
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu = self.actor(state_t)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
            act = dist.sample()
            lp = dist.log_prob(act)  # shape: [1, action_dim]
            logp = lp.sum(dim=-1).item()  # scalar
            val = self.critic(state_t).squeeze(-1).item()
        return act.cpu().numpy(), logp, val

    def evaluate(self, states, actions):
        """
        Evaluates log probabilities, entropy, and value estimates for given states and actions.

        Args:
            states (Tensor): Batch of states [T, state_dim]
            actions (Tensor): Batch of actions [T, action_dim]

        Returns:
            log_probs (Tensor): Log probabilities [T]
            entropy (Tensor): Mean entropy (scalar)
            values (Tensor): Value estimates [T]
        """
        mu = self.actor(states)
        vals = self.critic(states).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)

        logp = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(-1).mean()
        return logp, entropy, vals

    def update(self, ent_coef: Optional[float] = None, target_kl: float = 0.01, global_step: Optional[int] = None):
        """
        Performs PPO updates using collected rollout buffer.

        Args:
            ent_coef (float, optional): Entropy coefficient (default: base_ent_coef)
            target_kl (float): KL divergence target for early stopping
            global_step (int, optional): Global step for logging
        """
        coef = ent_coef if ent_coef is not None else self.base_ent_coef

        # 1) Stack rollout tensors
        states = torch.stack(self.buffer.states)    # [T, state_dim]
        actions = torch.stack(self.buffer.actions)  # [T, action_dim]
        old_lp = torch.stack(self.buffer.log_probs) # [T]

        # 2) Bootstrap the final value
        with torch.no_grad():
            last_s = self.buffer.states[-1]
            last_v = self.critic(last_s).squeeze(-1).item()

        # 3) Compute returns and advantages using GAE
        returns, advs = compute_gae(
            rewards=self.buffer.rewards,
            dones=self.buffer.dones,
            values=self.buffer.values + [last_v],
            gamma=self.gamma, lam=self.lam
        )
        returns = torch.FloatTensor(returns).to(self.device)
        advs = torch.FloatTensor(advs).to(self.device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # 4) Precompute old critic values for clipped loss
        with torch.no_grad():
            old_vals = self.critic(states).squeeze(-1)

        # 5) PPO multiple epochs
        csv_f = open(self.csv_path, "a", newline="")
        writer = csv.writer(csv_f)

        for epoch in range(1, self.K_epochs + 1):
            new_lp, entropy, vals = self.evaluate(states, actions)

            # Actor loss: clipped surrogate objective
            ratios = torch.exp(new_lp - old_lp)
            surr1 = ratios * advs
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advs
            loss_actor = -torch.min(surr1, surr2).mean()

            # Critic loss: clipped value loss
            uncl = (vals - returns).pow(2)
            clip_vals = old_vals + (vals - old_vals).clamp(-self.eps_clip, self.eps_clip)
            clip = (clip_vals - returns).pow(2)
            loss_critic = 0.5 * torch.max(uncl, clip).mean()

            # Total loss
            total_loss = loss_actor + loss_critic - coef * entropy

            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()

            # Compute KL divergence for early stopping
            with torch.no_grad():
                kl = (old_lp - new_lp).mean().item()

            # Log training metrics to CSV
            writer.writerow([
                global_step, epoch,
                loss_actor.item(),
                loss_critic.item(),
                entropy.item(),
                total_loss.item(),
                kl
            ])

            # Early stop if KL divergence is too large
            if kl > 1.5 * target_kl:
                break

        csv_f.close()
        self.buffer.clear()

    def save(self, filepath: str):
        """
        Saves the actor, critic, log_std, and optimizer states.

        Args:
            filepath (str): Path to save checkpoint
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'actor_state_dict':  self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'log_std':           self.log_std.detach().cpu(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"[PPOAgent] Model saved to {filepath}")

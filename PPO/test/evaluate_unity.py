import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import yaml
from envs.unity_env_wrapper import UnityEnvWrapper
from ppo.ppo_agent import PPOAgent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to config file"
    )
    p.add_argument(
        "--run_dir", 
        type=str, 
        default="saved_models/", 
        help="Directory where agent checkpoints live"
    )
    p.add_argument(
        "--episodes", 
        type=int, 
        default=5, 
        help="Number of episodes to evaluate"
    )
    return p.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    config = load_config(args.config)

    # pick up smoothing alpha from config (or fallback)
    alpha = float(
        config["train"].get("eval_smoothing_alpha", 0.85)
    )

    # create Unity env with graphics
    env = UnityEnvWrapper(
        config["env"]["unity_path"], 
        no_graphics=False
    )
    obs = env.reset(train_mode=False)

    n_agents    = obs.shape[0]
    state_size  = obs.shape[1]
    action_size = env.brain.vector_action_space_size

    # load each agent’s checkpoint
    agents = []
    for i in range(n_agents):
        agent = PPOAgent(state_size, action_size, config)
        ckpt   = os.path.join(args.run_dir, f"agent{i}_best.pth")
        checkpoint = torch.load(ckpt, map_location=agent.device)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.log_std = checkpoint["log_std"].to(agent.device)
        agent.actor.to(agent.device)
        agent.critic.to(agent.device)
        agents.append(agent)

    print(f"Loaded {n_agents} agents; evaluating {args.episodes} eps with α={alpha}")

    for ep in range(1, args.episodes+1):
        obs    = env.reset(train_mode=False)
        dones  = [False]*n_agents
        scores = np.zeros(n_agents, dtype=np.float32)

        # initialize smoothing
        prev_actions = [
            np.zeros(action_size, dtype=np.float32) 
            for _ in range(n_agents)
        ]

        while not any(dones):
            # sample raw actions
            raw_actions = []
            for i, agent in enumerate(agents):
                a, _lp, _v = agent.select_action(obs[i])
                raw_actions.append(a)

            # exponential smoothing
            smooth_actions = []
            for i in range(n_agents):
                a_s = alpha * raw_actions[i] + (1 - alpha) * prev_actions[i]
                smooth_actions.append(a_s)
            prev_actions = smooth_actions

            # step
            next_obs, rewards, dones, _ = env.step(smooth_actions)
            obs = next_obs
            scores += np.array(rewards, dtype=np.float32)

        print(
            f"[Eval {ep}] " 
            f"Scores: {scores} | "
            f"Max: {scores.max():.3f} | "
            f"Avg: {scores.mean():.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()

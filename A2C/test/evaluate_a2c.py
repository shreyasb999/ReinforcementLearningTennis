import argparse
import sys
import os
import yaml
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.unity_env_wrapper import UnityEnvWrapper
from a2c.a2c_agent import A2CAgent 

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
        help="Directory where A2C checkpoints live"
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate"
    )
    return p.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    config = load_config(args.config)

    # smoothing factor α for exponential action smoothing at eval time
    alpha = float(
        config["train"].get("eval_smoothing_alpha", 0.85)
    )

    # create Unity env (with graphics so you can watch)
    env = UnityEnvWrapper(
        config["env"]["unity_path"],
        no_graphics=False
    )
    obs = env.reset(train_mode=False)

    n_agents    = obs.shape[0]
    state_size  = obs.shape[1]
    action_size = env.brain.vector_action_space_size

    # load trained A2C agents
    agents = []
    for i in range(n_agents):
        agent = A2CAgent(state_size, action_size, config)
        ckpt   = os.path.join(args.run_dir, f"agent{i}_best_a2c.pth")
        checkpoint = torch.load(ckpt, map_location=agent.device)
        agent.shared.load_state_dict(checkpoint["shared_state_dict"])
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.log_std.data.copy_(checkpoint["log_std"])
        agents.append(agent)

    print(f"Loaded {n_agents} A2C agents; evaluating {args.episodes} episodes with α={alpha}")

    for ep in range(1, args.episodes+1):
        obs    = env.reset(train_mode=False)
        dones  = [False]*n_agents
        scores = np.zeros(n_agents, dtype=np.float32)

        # initialize exponential smoothing
        prev_actions = [
            np.zeros(action_size, dtype=np.float32)
            for _ in range(n_agents)
        ]

        while not any(dones):
            raw_actions = []
            # sample actions from each agent
            for i, agent in enumerate(agents):
                a, _logp, _val = agent.select_action(obs[i])
                raw_actions.append(a)

            # apply exponential smoothing
            smooth_actions = []
            for i in range(n_agents):
                a_s = alpha * raw_actions[i] + (1 - alpha) * prev_actions[i]
                smooth_actions.append(a_s)
            prev_actions = smooth_actions

            # step the Unity env
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

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from envs.unity_env_wrapper import UnityEnvWrapper
from ppo.ppo_agent import PPOAgent
import csv
import torch
import datetime


def train(config):
    # --- CSV logging setup ---
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")

    filename = f"training_{date_str}.csv"
    csv_path = config["train"].get("train_logs_dir", "logs/train")
    csv_path = os.path.join(csv_path, filename)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "volleys", "max_reward", "avg_reward", "ent_coef"])

    # --- Pull configurations from the config dictionary ---
    env_path      = config["env"]["unity_path"]
    total_episodes= int(config["train"]["total_episodes"])
    timesteps     = int(config["train"]["timesteps"])
    init_ent_coef = float(config["ppo"]["initial_entropy_coef"])
    reward_scale  = float(config["ppo"]["reward_scale"])

    # --- Create Unity environment (with graphics enabled) ---
    env = UnityEnvWrapper(env_path, no_graphics=False)
    obs = env.reset(train_mode=True)
    n_agents    = obs.shape[0]  # Number of agents (e.g., tennis players)
    state_size  = obs.shape[1]  # State dimensionality
    action_size = env.brain.vector_action_space_size  # Action space dimensionality

    # --- Initialize one PPOAgent per agent in the environment ---
    agents = [PPOAgent(state_size, action_size, config) for _ in range(n_agents)]

    # --- Setup model saving ---
    save_dir = config["train"].get("save_dir", "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_avg = -float('inf')  # Track best average reward for model saving

    for ep in range(1, total_episodes + 1):
        # Linearly anneal entropy coefficient over training
        ent_coef = init_ent_coef * max(0.0, 1 - ep / total_episodes)

        # Reset environment and clear each agent's buffer
        obs = env.reset(train_mode=True)
        for a in agents:
            a.buffer.clear()

        # Initialize per-episode metrics
        rewards_sum = np.zeros(n_agents, dtype=np.float32)
        volley_cnt  = np.zeros(n_agents, dtype=int)

        steps = 0
        while steps < timesteps:
            # 1) Each agent selects an action, log probability, and value
            actions, logps, vals = [], [], []
            for i, a in enumerate(agents):
                act, logp, val = a.select_action(obs[i])
                actions.append(act)
                logps.append(logp)
                vals.append(val)

            # 2) Step the environment using selected actions
            next_obs, rewards, dones, _ = env.step(actions)
            rewards = np.array(rewards, dtype=np.float32)

            # 3) Accumulate rewards and volley counts for logging
            rewards_sum += rewards
            volley_cnt  += (rewards > 0).astype(int)

            # 4) Store experience in each agent's buffer
            for i, a in enumerate(agents):
                a.buffer.add(
                    state    = obs[i],
                    action   = actions[i],
                    log_prob = logps[i],
                    reward   = float(rewards[i]) * reward_scale,
                    done     = dones[i],
                    value    = vals[i],
                    device   = a.device
                )

            # 5) Immediate environment reset if any agent is done
            if any(dones):
                obs = env.reset(train_mode=True)
            else:
                obs = next_obs

            steps += 1

        # --- After collecting exactly `timesteps` steps ---
        # Store final states for GAE (Generalized Advantage Estimation) bootstrapping
        for i, a in enumerate(agents):
            a.buffer.last_state = obs[i]

        # Perform PPO update for each agent
        for a in agents:
            a.update(ent_coef=ent_coef, global_step=ep)

        # --- Logging after each episode ---
        mx  = rewards_sum.max()   # Maximum reward across agents
        avg = rewards_sum.mean()  # Average reward across agents
        volleys = volley_cnt.max()  # Maximum volley count
        print(f"Episode {ep:4d} | Volleys {volleys:3d} | MaxRwd {mx:.3f} | AvgRwd {avg:.3f} | Ent={ent_coef:.4f}")
        csv_writer.writerow([ep, volleys, mx, avg, ent_coef])

        # Save models if there is an improvement in average reward
        if avg > best_avg:
            best_avg = avg
            for i, a in enumerate(agents):
                path = os.path.join(save_dir, f"agent{i}_best.pth")
                a.save(path)

        # Clear any leftover GPU memory after episode
        torch.cuda.empty_cache()

    # --- Clean-up: close environment and CSV file ---
    env.close()
    csv_file.close()
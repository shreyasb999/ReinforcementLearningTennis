import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.unity_env_wrapper import UnityEnvWrapper
from a2c.a2c_agent import A2CAgent
import numpy as np
import csv
import torch
import datetime

def train(config):
    # CSV & logging setup
    now      = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")
    save_dir = config["train"].get("save_dir", "saved_models/a2c")
    os.makedirs(save_dir, exist_ok=True)

    csv_filename = f"a2c_training_{date_str}.csv"
    csv_path = os.path.join(config["train"].get("train_logs_dir", "logs/train"), csv_filename)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode","volleys","max_reward","avg_reward","ent_coef"])

    # config pulls
    env_path       = config["env"]["unity_path"]
    total_episodes = int(config["train"]["total_episodes"])
    timesteps      = int(config["train"]["timesteps"])
    init_ent_coef  = float(config["ppo"]["initial_entropy_coef"])
    reward_scale   = float(config["ppo"]["reward_scale"])

    # init Unity env
    env = UnityEnvWrapper(env_path, no_graphics=False)
    obs = env.reset(train_mode=True)
    n_agents    = obs.shape[0]
    state_size  = obs.shape[1]
    action_size = env.brain.vector_action_space_size

    # create A2C agents
    agents = [A2CAgent(state_size, action_size, config) for _ in range(n_agents)]
    best_avg = -float("inf")

    for ep in range(1, total_episodes+1):
        ent_coef = init_ent_coef * max(0.0, 1 - ep/total_episodes)
        obs = env.reset(train_mode=True)
        for a in agents:
            a.buffer.clear()

        rewards_sum = np.zeros(n_agents, dtype=np.float32)
        volley_cnt  = np.zeros(n_agents, dtype=int)
        steps = 0

        while steps < timesteps:
            actions, logps, vals = [], [], []
            for i, a in enumerate(agents):
                act, lp, val = a.select_action(obs[i])
                actions.append(act)
                logps.append(lp)
                vals.append(val)

            next_obs, rewards, dones, _ = env.step(actions)
            rewards = np.array(rewards, dtype=np.float32)
            rewards_sum += rewards
            volley_cnt  += (rewards > 0).astype(int)

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

            obs = env.reset(train_mode=True) if any(dones) else next_obs
            steps += 1

        # update each agent
        for a in agents:
            a.update()

        # episode stats
        mx    = float(rewards_sum.max())
        avg   = float(rewards_sum.mean())
        volleys = int(volley_cnt.max())
        print(f"Episode {ep:4d} | Volleys {volleys:3d} | MaxRwd {mx:.3f} | AvgRwd {avg:.3f} | Ent={ent_coef:.4f}")
        csv_writer.writerow([ep, volleys, mx, avg, ent_coef])

        # save best
        if avg > best_avg:
            best_avg = avg
            for i, a in enumerate(agents):
                path = os.path.join(save_dir, f"agent{i}_best_a2c.pth")
                a.save(path)

        torch.cuda.empty_cache()

    env.close()
    csv_file.close()

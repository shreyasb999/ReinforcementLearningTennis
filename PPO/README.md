
##### Shreyas Bhavsar - April 2025

# Proximal Policy Optimization (PPO) - Unity Tennis Multi-Agent
This repository implements a Proximal Policy Optimization (PPO) algorithm to train two agents in Unity’s Tennis environment (multi-agent). It is fully modular and configurable via config.yaml.

## Directory Structure
```
ppo_project/
│
├── envs/
│   └── unity_env_wrapper.py   # Gym-style wrapper around Unity Tennis
│
├── models/
│   └── actor_critic.py        # Shared network definitions
│
├── ppo/
│   ├── buffer.py              # RolloutBuffer to store trajectories
│   ├── ppo_agent.py           # PPOAgent: actor, critic, update logic
│   └── utils.py               # compute_gae utility
│
├── train/
│   └── train_unity.py         # Collect rollouts, update agents, logging & model saving
│
├── test/
│   └── evaluate_unity.py      # Run trained agents in Unity Tennis & evaluate performance
│
├── saved_models/              # Saved `.pth` checkpoints
│
├── logs/
│   ├── train/                 # Per-episode CSV logs
│   └── agents/                # Per-agent PPO-update CSV logs
│
├── main.py                    # Entry point: parses config & launches training
└── config.yaml                # All hyperparameters & paths
```


## Configuration

All hyperparameters and file paths live in config.yaml.
Just edit config.yaml—no need to change any code files.


## How to Run

1. Place your Unity Tennis executable path in config.yaml → env.unity_path

```
env:
  unity_path: "/path/to/Tennis.exe"
```

2. Launch training:

```
python main.py 
```

A Unity window will open, and agents will learn in real time.

3. Check logs:

	* Per-episode CSVs in logs/train/

	* Per-agent update-step CSVs in logs/agents/

	* Model checkpoints in saved_models/
	
	
## Results & Visualization

* Episode logs record max/avg rewards and entropy coefficient.

* Agent logs record actor/critic losses, entropy, KL per update epoch.

* You can plot CSVs to monitor training curves.


## Evaluation Script

After training, you can evaluate your saved Tennis agents with optional action–smoothing and visualize their performance in the Unity window.

### Configuration

Add the following field to your `config.yaml` under the `train` section (defaults to 0.85 if omitted):

```
train:
  eval_smoothing_alpha: 0.85   

```

### Usage

```
python evaluate_unity.py 
```

or by passing following arguments:

```
python evaluate_unity.py  --config config.yaml  --run_dir saved_models/  --episodes 10
```

* --config :
	Path to your config.yaml.

* --run_dir : 
	Directory containing your saved agent checkpoints (agent0_best.pth, agent1_best.pth, …).

* --episodes :
	Number of full episodes to run during evaluation.
	

## Next Steps

* Checkpoint loading & evaluation: add a script to load saved .pth and evaluate or record videos.

* Hyperparameter search: tweak config.yaml to improve performance.

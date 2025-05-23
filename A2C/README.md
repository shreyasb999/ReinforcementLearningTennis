
##### Shreyas Bhavsar - April 2025

## Advantage Actor-Critic (A2C) - Unity Tennis Multi-Agent
This repository implements an Advantage Actor-Critic (A2C) algorithm to train two agents in Unity’s Tennis environment (multi-agent). It is fully modular and configurable via `config.yaml`.


### Directory Structure
```
ppo_project/
│
├── envs/
│   └── unity_env_wrapper.py   # Gym-style wrapper around Unity Tennis
│
├── a2c/
│   ├── buffer.py              # RolloutBuffer to store trajectories
│   ├── a2c_agent.py           # A2CAgent: actor, critic, update logic
│   └── utils.py               # compute_gae utility
│
├── train/
│   └── train_a2c.py           # Collect rollouts, update agents, logging & model saving
│
├── saved_models/              # Saved `.pth` checkpoints
│
├── logs/
│   └── train/                 # Per-episode CSV logs
│
├── main.py                    # Entry point: parses config & launches training
└── config.yaml                # All hyperparameters & paths
```


### Configuration

All hyperparameters and file paths live in config.yaml.
Just edit config.yaml—no need to change any code files.


### How to Run

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

	* Model checkpoints in saved_models/
	
	
### Results & Visualization

* Episode logs record max/avg rewards and entropy coefficient.

* You can plot CSVs to monitor training curves.


### Evaluation Script

After training, you can evaluate your saved Tennis agents with optional action–smoothing and visualize their performance in the Unity window.

#### Configuration

Add the following field to your `config.yaml` under the `train` section (defaults to 0.85 if omitted):

```
train:
  eval_smoothing_alpha: 0.85   

```

#### Usage

```
python evaluate_a2c.py 
```

or by passing following arguments:

```
python evaluate_a2c.py  --config config.yaml  --run_dir saved_models/  --episodes 10
```

* --config
	Path to your config.yaml.

* --run_dir
	Directory containing your saved agent checkpoints (agent0_best.pth, agent1_best.pth, …).

* --episodes
	Number of full episodes to run during evaluation.
	

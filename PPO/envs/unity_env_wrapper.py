from unityagents import UnityEnvironment
import numpy as np

class UnityEnvWrapper:
    """
    Wraps a Unity ML-Agents multi-agent environment with a Gym-like API.
      - reset() → obs (np.ndarray, shape [n_agents, state_dim])
      - step(actions) → (next_obs, rewards, dones, info)
      - exposes num_agents, state_size, action_size, and brain
    """

    def __init__(self, env_path: str, no_graphics: bool = False):
        # Launch (or connect to) the Unity executable
        self.env = UnityEnvironment(file_name=env_path, no_graphics=no_graphics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Will be set on reset()
        self.num_agents = None
        self.state_size = None
        self.action_size = None

    def reset(self, train_mode: bool = True):
        """
        Resets the Unity environment.

        Returns:
            obs: np.ndarray, shape [n_agents, state_dim], dtype float32
        """
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]

        self.num_agents  = len(env_info.agents)
        # each observation is a list/array
        self.state_size  = len(env_info.vector_observations[0])
        self.action_size = self.brain.vector_action_space_size

        obs = np.array(env_info.vector_observations, dtype=np.float32)
        return obs

    def step(self, actions):
        """
        Steps the environment with a list/array of actions (one per agent).

        Returns:
            next_obs: np.ndarray [n_agents, state_dim], float32
            rewards:  np.ndarray [n_agents], float32
            dones:    np.ndarray [n_agents], bool
            info:     dict (empty for now)
        """
        env_info = self.env.step(actions)[self.brain_name]
        next_obs = np.array(env_info.vector_observations, dtype=np.float32)
        rewards  = np.array(env_info.rewards, dtype=np.float32)
        dones    = np.array(env_info.local_done, dtype=np.bool_)
        return next_obs, rewards, dones, {}

    def close(self):
        """Close the Unity process."""
        try:
            self.env.close()
        except Exception:
            pass
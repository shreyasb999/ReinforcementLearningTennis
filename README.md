# Tennis Reinforcement Learning

This project uses Unity's Machine Learning Agents (ML-Agents) to train two reinforcement learning agents to play tennis against each other. The environment is set up using Unity and Python, and the agents are trained using PPO and A2C.

## Setup Instructions

# 1. Install the Anaconda Navigator
Download from https://www.anaconda.com/products/navigator, following the guidelines

# 2. Create a Python 3.6 environment
Run this command in the Anaconda Prompt 
```bash
conda create --name RL python=3.6
```

# 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Note: If you are on Windows and you get the error "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)" occurs, refer to duhgrando's answer on this thread:
https://github.com/udacity/deep-reinforcement-learning/issues/13

# 4. Set Up Unity Environment

Download the Tennis Environment (Udacity's modified version) build and place it in the Tennis folder within the project directory. Depending on your operating system:

Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip

Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip

Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip

Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

(Note the PPO and A2C code was done on a windows machine, so the Unity file present in the ```PPO``` and ```A2C``` folders is compatible with Windows OS and should be replaced with the respective file as supported by your machine)


# Running the Code
Once these steps have been followed, open the respective PPO and A2C folders and follow the instructions there on how to run the code.


# Additional Notes and Credits

This code was a part of a group project for Reinforcement Learning module during my MSc Data Science Course.

The PPO code is developed by Shreyas Bhavsar and Georgios Kotnis.

The A2C code is written by Shreyas Bhavsar.

To check out the entire project, which compares PPO vs DDPG vs A2C - you can check out our original project repo - 
https://github.com/AkshayaJeyaram/TennisReinforcementLearning.git


# Application 2: Ballkicking using RL

This repository contains a ball kicking experiment with reinforcement learning (RL) based on our equivalent model implemented in **MuJoCo**.

## ⌘ Results
![PID Convergence](figure/joint_trajectory.png)
Ball kicking is achieved through RL training. We implement the pipeline with stable_baselines3 and gymnasium.

## 📦 Project Structure

- `kick_env.py`: Environment definition for the ball kicking task.
- `train_kick.py`: Main training script for the RL agent.
- `evaluate.py`: Script to evaluate the trained model.
- `data/`: Tensorboard log file. 
- `figure/`: Plot output directory.
- `video/`: Video output directory.

## 🚀 Requirements
Same as the whole work directory.

## ▶️ How to Run the Experiment

1. **Run Training**
```bash
python src/application/RL_BallKicking/scripts/train_kick.py
```
- Output: Tensorboard log file (`data/log`), trained model saved in `data/model`.

2. **Run Evaluation**
```bash
python src/application/RL_BallKicking/scripts/evaluate.py
```
- Output: Video output in `video/`.

## ⚙️ Key Features
- **Reinforcement Learning**: Uses stable_baselines3 for training a PPO agent.
- **Environment Setup**: Implements a custom gymnasium environment for the ball kicking task.
- **Result Analysis**: Automatically generates evaluation metrics and videos for review.

## 📂 Output Files
- `data/log/ppo_kick_tensorboard/PPO_1/*`: Tensorboard log files for training.
- `data/model/ppo_kick_model.zip`: Trained PPO model.
- `video/kick_evaluation.mp4`: Video of the evaluation run.

## 📜 License
MIT License

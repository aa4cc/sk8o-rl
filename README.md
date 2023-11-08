# Reinforcement-Learning-based Control for a Two-Wheeled-Legged Balancing Robot using a Simple Linear Model
This repo contains the code for our paper of the same name. You can install the requirements using the `requirements.txt` file or build a Singularity container based on the `sk8o-rl.def` file.

## Example usage
After installing all the dependencies, you can start the training e.g. by
```
python training/train.py optimizer.lr=1e-4 optimizer.total_timesteps=2e6
```
The code supports logging via Weights & Biases (use `wandb=true` to enable) and configuration via hydra (see `training/config`).

## The models
The models used for training and verification are available [via PyPI](https://pypi.org/project/sk8o-sim/). 

# RL course final project: A Comparison of Double Deep Q-Networks and Asynchronous Advantage Actor-Critic

In this project, we explore how Reinforcement Learning (RL) can tackle the classic Nintendo game Mario Bros NES. Using an [https://pypi.org/project/gym-super-mario-bros/](gym-super-mario-bros), we implemented and compared two prominent algorithms: Double Deep Q-Learning (DDQN) and Advantage Actor Critic (A2C).

Install the dependencies (we ran the code on an python 3.11 environment):
```pip install -r requirements.txt```

To train the agents:

```python dqn/train.py```
```python a2c/train.py```

The checkpoints and the metrics will be saved in `checkpoint/yyyy-mm-ddThh-mm-ss`

To render the game once an agent is trained:
```python dqn/render.py --checkpoints path_to_chpkt```
```python a2c/render.py --checkpoints path_to_chpkt```
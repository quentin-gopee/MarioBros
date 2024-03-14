import torch
from pathlib import Path
import datetime
from mario import Mario
from env import create_env
from metric import MetricLogger


def main():
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    print(f"Save directory: {save_dir}")

    print("Creating environment...")
    env = create_env()

    mario = Mario(state_dim=(4, 84, 84),
                  action_dim=env.action_space.n,
                  save_dir=save_dir,
                  save_every=5e5)

    logger = MetricLogger(save_dir)

    episodes = 40000

    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)


if __name__ == "__main__":
    main()
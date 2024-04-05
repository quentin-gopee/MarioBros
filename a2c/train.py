import torch
from pathlib import Path
import datetime
from agent import ActorCriticMario
from env import create_env
from metric import MetricLogger
import argparse

def main():
    # parser
    parser = argparse.ArgumentParser(description="Train Mario")
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint file")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    print(f"Save directory: {save_dir}")

    print("Creating environment...")
    env = create_env()

    mario = ActorCriticMario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, save_every=20000)
    
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        check = torch.load(args.checkpoint, map_location=device)
        mario.net.load_state_dict(check["model"])
        print(f"Model loaded from {args.checkpoint}")

    logger = MetricLogger(save_dir)

    episodes = 100
    local_steps = 50

    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:
            mario.clear_cache()

            for _ in range(local_steps):
                # Run agent on the state
                action, value, log_prob, entropy = mario.act(state)
                
                # Agent performs action
                state, reward, done, trunc, info = env.step(action)

                # Log
                logger.log_step(reward, info["x_pos"])

                # Remember
                mario.cache(value, log_prob, reward, entropy)

                # Check if end of game
                if done or info["flag_get"]:
                    break

            q, actor_loss, critic_loss, entropy_loss, total_loss = mario.learn(state, done)

            logger.log_learn(q, actor_loss, critic_loss, entropy_loss, total_loss, info['x_pos'])   

            if done or info["flag_get"]:
                break

        logger.log_episode()
        print(f"Episode {e} - Step {mario.curr_step}")

        if (e % 5 == 0) or (e == episodes - 1):
            logger.record(episode=e, step=mario.curr_step)

if __name__ == "__main__":
    main()
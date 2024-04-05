import torch
from pathlib import Path
from agent import Mario
from env import create_env
import argparse


def main():
    # parser for checkpoint
    parser = argparse.ArgumentParser(description="Play Mario")
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint file")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    print("Creating environment...")
    env = create_env(render=True)
    print(env.action_space.n)

    save_dir = ""

    mario = Mario(state_dim=(4, 84, 84),
                  action_dim=env.action_space.n,
                  save_dir=save_dir,
                  save_every=5e5)
    
    checkpoint = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check = torch.load(checkpoint, map_location=device)
    mario.net.load_state_dict(check["model"])
    print(f"Model loaded from {checkpoint}")

    while True:

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)
            
            env.render()

            # Check if end of game
            if done or info["flag_get"]:
                break


if __name__ == "__main__":
    main()
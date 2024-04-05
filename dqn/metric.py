import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd

class MetricLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.save_log = save_dir / "log"
        self.metrics_file = save_dir / "metrics.csv"

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        # Check if metrics.csv exists, if not, create it with header
        pd.DataFrame(columns=['Episode', 'Step', 'Epsilon', 'MeanReward', 'MeanLength', 'MeanLoss', 'MeanQValue', 'TimeDelta', 'Time']).to_csv(self.metrics_file, index=False)

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"
        self.ep_x_pos_plot = save_dir / "x_pos_plot.jpg"
        self.max_x_pos_plot = save_dir / "max_x_pos_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_avg_x_pos = []
        self.max_x_pos = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        self.moving_avg_ep_x_pos = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

        # Episode data
        self.episode_data = pd.DataFrame(columns=['Episode', 'Step', 'Epsilon', 'MeanReward', 'MeanLength', 'MeanLoss', 'MeanQValue', 'TimeDelta', 'Time', 'MeanXPos', 'MaxXPos'])

    def log_step(self, reward, loss, q, x_pos):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        # we sum the x position of the agent within a episode
        self.curr_x_pos.append(x_pos)
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        
        ep_avg_x = np.round(np.mean(self.curr_x_pos), 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        # we add avergae x position to the list
        self.ep_avg_x_pos.append(ep_avg_x)
        self.max_x_pos.append(max(self.curr_x_pos) if self.curr_x_pos else 0)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        self.curr_x_pos = []

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        mean_ep_x_pos = np.round(np.mean(self.ep_avg_x_pos[-100:]), 3) 
        max_x_pos = self.max_x_pos[-1]
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)
        self.moving_avg_ep_x_pos.append(mean_ep_x_pos)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Mean X Pos {mean_ep_x_pos} - "
            f"Max X Pos {max_x_pos} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        # Update the episode_data DataFrame
        self.episode_data.loc[len(self.episode_data)] = {
            'Episode': episode,
            'Step': step,
            'Epsilon': epsilon,
            'MeanReward': mean_ep_reward,
            'MeanLength': mean_ep_length,
            'MeanLoss': mean_ep_loss,
            'MeanQValue': mean_ep_q,
            'TimeDelta': time_since_last_record,
            "MeanXPos": mean_ep_x_pos,
            "MaxXPos": max(self.curr_x_pos) if self.curr_x_pos else 0,
            'Time': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        }

        # Save the DataFrame to CSV
        self.episode_data.to_csv(self.metrics_file, index=False)
        
        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards", "ep_x_pos"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))
            
        plt.clf()
        plt.plot(self.max_x_pos, label="Max X Pos")
        plt.legend()
        plt.savefig(getattr(self, f"max_x_pos_plot"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from model import ActorCritic, DeepCnnActorCriticNetwork, BaseActorCriticNetwork  # Assurez-vous que ActorCritic est correctement importé
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
# from tensordict import TensorDict  # Assure-toi que ceci est nécessaire ou disponible dans ton environnement
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from model import ActorCritic, DeepCnnActorCriticNetwork, BaseActorCriticNetwork  # Vérifie les imports

class ActorCriticMario:
    def __init__(self, state_dim, action_dim, save_dir, save_every):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.save_every = save_every

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = DeepCnnActorCriticNetwork(self.state_dim, self.action_dim).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)

        self.curr_step = 0

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

        self.batch_size = 32
        self.gamma = 0.9 # discount factor
        self.tau = 1.0 # gae factor
        self.beta = 0.01 # entropy coefficient

    
    def act(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            policy, value = self.net(state)
        prob = F.softmax(policy, dim=1)
        log_prob = F.log_softmax(policy, dim=1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        action = Categorical(prob).sample().item()

        self.curr_step += 1

        return action, value, log_prob[0, action], entropy
    
    
    def cache(self, value, log_prob, reward, entropy):
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.entropies.append(entropy)


    def clear_cache(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
    
    
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def learn(self, state, done):
        if self.curr_step % self.save_every == 0:
            self.save()

        R = torch.zeros((1, 1), dtype=torch.float)
        R.to(self.device)
        if not done:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            _, R = self.net(state)

        gae = torch.zeros((1, 1), dtype=torch.float)
        gae.to(self.device)
        actor_loss = torch.zeros((1, 1), dtype=torch.float, requires_grad=True)
        critic_loss = torch.zeros((1, 1), dtype=torch.float, requires_grad=True)
        entropy_loss = torch.zeros((1, 1), dtype=torch.float, requires_grad=True)
        next_value = R

        for value, log_policy, reward, entropy in list(zip(self.values, self.log_probs, self.rewards, self.entropies))[::-1]:
            gae = gae * self.gamma * self.tau
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * self.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - self.beta * entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return np.mean(self.values), actor_loss.detach(), critic_loss.detach(), entropy_loss.detach(), total_loss.detach()


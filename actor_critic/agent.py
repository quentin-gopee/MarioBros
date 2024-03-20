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

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=self.device))
        self.batch_size = 32

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)

        self.burnin = 1e4  # Expériences min. avant entraînement
        self.learn_every = 3  # Nombre d'expériences entre les mises à jour de Q_online
        self.sync_every = 1e4  # Supposons que c'est pour synchroniser les réseaux target et online
        self.entropy_coef = 0.01
    
    def act(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = self.net(state)
        probabilities = F.softmax(policy, dim=-1).cpu().numpy()
        action_idx = np.random.choice(self.action_dim, p=probabilities.squeeze())
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))
    
    def prepare_training_data(self):
        batch = self.memory.sample(self.batch_size)
        states = batch['state'].to(self.device).float()
        next_states = batch['next_state'].to(self.device).float()
        actions = torch.tensor(batch['action'], device=self.device).long()
        rewards = torch.tensor(batch['reward'], device=self.device).float()
        dones = torch.tensor(batch['done'], device=self.device).float()
        return states, next_states, actions, rewards, dones
    
    def learn(self):
        states, next_states, actions, rewards, dones = self.prepare_training_data()
        
        # Obtient les valeurs actuelles et prévues
        current_policy, current_value = self.net(states)
        _, next_value = self.net(next_states)
        
        # Calcule les avantages et les TD errors
        next_value_detached = next_value.detach()
        td_target = rewards + self.gamma * next_value_detached * (1 - dones)
        td_error = td_target - current_value.squeeze()
        
        # Calcul de l'entropie pour encourager l'exploration
        log_probs = F.log_softmax(current_policy, dim=1)
        probs = F.softmax(current_policy, dim=1)
        entropy_loss = -(log_probs * probs).sum(-1).mean()

        # Objectif de l'acteur
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        action_log_probs = (log_probs * actions_onehot).sum(-1)
        actor_loss = -(action_log_probs * td_error.detach()).mean()
        
        # Objectif du critique
        critic_loss = td_error.pow(2).mean()

        # Mise à jour globale
        loss = actor_loss + critic_loss - self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return td_error.mean().item(),loss.item()


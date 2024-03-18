import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from model import ActorCritic, DeepCnnActorCriticNetwork, BaseActorCriticNetwork  # Assurez-vous que ActorCritic est correctement importé

class ActorCriticMario:
    def __init__(self, state_dim, action_dim, save_dir, save_every):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.save_every = save_every  # no. of experiences between saving Mario Net

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = BaseActorCriticNetwork(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4 

        # Add additional variables
        self.use_standardization = True  # Example value, modify as needed
        self.stable_eps = 1e-8  # Example value, modify as needed
        self.lamb = 0.75  # Example value, modify as needed
        self.beta = 0.5  # Example value, modify as needed
        self.clip_grad_norm = 1.0  # Example value, modify as needed
        self.entropy_coef = 0.01  # Example value, modify as needed
    
    def act(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        print(state.shape)
        state = torch.tensor(state, device=self.device)
        policy, value = self.net(state)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(policy)

        return action
    
    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)
    
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
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

    def forward_transition(self, state, next_state):
        state = torch.from_numpy(state).to(self.device)
        state = state.float()
        policy, value = self.net(state)

        next_state = torch.from_numpy(next_state).to(self.device)
        next_state = next_state.float()
        _, next_value = self.net(next_state)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value, policy
    
    def learn(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch):
        with torch.no_grad():
            s_batch = torch.FloatTensor(s_batch).to(self.device)
            next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
            target_batch = torch.FloatTensor(target_batch).to(self.device)
            y_batch = torch.LongTensor(y_batch).to(self.device)
            adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        if self.use_standardization:
            adv_batch = (adv_batch - adv_batch.mean()) / \
                (adv_batch.std() + self.stable_eps)

        # Calcul de l'avantage
        with torch.no_grad():
            _, next_value = self.net(next_s_batch)
            target_values = target_batch + self.gamma * (1 - next_s_batch) * next_value

        # Calcul de l'erreur TD
        values = self.net(s_batch)[1]
        td_error = target_values - values

        # Calcul des logits et des valeurs d'état-action
        policy, value = self.net(s_batch)
        log_probs = F.log_softmax(policy, dim=-1)
        actions_onehot = F.one_hot(y_batch, num_classes=self.action_dim)
        action_log_probs = torch.sum(log_probs * actions_onehot, dim=-1)

        # Calcul de l'objectif de l'acteur
        actor_loss = -torch.mean(action_log_probs * td_error.detach())

        # Calcul de l'objectif du critique
        critic_loss = torch.mean(td_error ** 2)

        # Mise à jour des poids
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

        return torch.mean(td_error).item(), loss.item()

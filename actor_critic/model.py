import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        c, h, w = num_inputs

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(c, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.actor_linear(x.view(x.size(0), -1)), self.critic_linear(x.view(x.size(0), -1))
    
    


class BaseActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseActorCriticNetwork, self).__init__()
        c, h, w = input_size

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        linear = nn.Linear

        self.feature = nn.Sequential(
            linear(c, 128),
            nn.ReLU(),
            linear(128, 128),
            nn.ReLU()
        )
        self.actor = linear(128, output_size)
        self.critic = linear(128, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

class DeepCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepCnnActorCriticNetwork, self).__init__()
        c, h, w = input_size

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")


        linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
            linear(50176, 512),
            nn.ReLU()
        )
        self.actor = linear(512, output_size)
        self.critic = linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

if __name__ == "__main__":
    model = ActorCritic(4, 14)
    print(model)
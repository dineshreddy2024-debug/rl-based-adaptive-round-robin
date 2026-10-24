import torch
import torch.nn as nn

from .ddqn import DDQNAgent


class DuelingNet(nn.Module):
    def __init__(self, n_state, n_actions):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # State-value stream
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        features = self.feature(x)

        value = self.value(features)
        advantage = self.advantage(features)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(
            dim=1,
            keepdim=True
        )

        return q_values


class D3QNAgent(DDQNAgent):

    def __init__(
        self,
        n_state,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        memory_size=10000
    ):
        # Initialize DDQN components
        super().__init__(
            n_state=n_state,
            n_actions=n_actions,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            #memory_size=memory_size
            )

        # Replace normal DQN networks with dueling networks
        self.net = DuelingNet(
            n_state,
            n_actions
        ).to(self.device)

        self.target = DuelingNet(
            n_state,
            n_actions
        ).to(self.device)

        # Initially target = online network
        self.target.load_state_dict(
            self.net.state_dict()
        )

        # New optimizer because the network architecture changed
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=lr
        )
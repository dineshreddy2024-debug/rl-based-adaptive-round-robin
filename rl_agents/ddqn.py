import random
import numpy as np
import torch
import torch.nn.functional as F

from .dqn import DQNAgent


class DDQNAgent(DQNAgent):
    def __init__(
        self,
        n_state,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05
    ):
              super().__init__(
                    n_state=n_state,
                    n_actions=n_actions,
                    lr=lr,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_min=epsilon_min
                    )
              self.optimizer = torch.optim.Adam(
                    self.net.parameters(),
                    lr=lr
                    )

    def train_episode(self, env, batch_size=64):

        state, _ = env.reset()

        done = False
        total_reward = 0.0
        losses = []

        while not done:

            # Select action using the online network
            action = self.act(state, training=True)

            next_state, reward, done, truncated, info = env.step(action)

            total_reward += reward

            self.memory.append(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
            )

            state = next_state

            # ------------------------------------------------
            # Double DQN update
            # ------------------------------------------------

            if len(self.memory) >= batch_size:

                batch = random.sample(
                    self.memory,
                    batch_size
                )

                states, actions, rewards, next_states, dones = zip(
                    *batch
                )

                states = torch.tensor(
                    np.asarray(states),
                    dtype=torch.float32,
                    device=self.device
                )

                actions = torch.tensor(
                    actions,
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(1)

                rewards = torch.tensor(
                    rewards,
                    dtype=torch.float32,
                    device=self.device
                )

                next_states = torch.tensor(
                    np.asarray(next_states),
                    dtype=torch.float32,
                    device=self.device
                )

                dones = torch.tensor(
                    dones,
                    dtype=torch.float32,
                    device=self.device
                )

                # Q(s,a) from online network
                current_q = self.net(states).gather(
                    1,
                    actions
                ).squeeze(1)

                # ------------------------------------------------
                # DOUBLE DQN:
                #
                # 1. Online network chooses the best action
                # 2. Target network evaluates that action
                # ------------------------------------------------

                with torch.no_grad():

                    next_actions = self.net(
                        next_states
                    ).argmax(
                        dim=1,
                        keepdim=True
                    )

                    next_q = self.target(
                        next_states
                    ).gather(
                        1,
                        next_actions
                    ).squeeze(1)

                    target_q = rewards + (
                        self.gamma *
                        next_q *
                        (1.0 - dones)
                    )

                # ------------------------------------------------
                # Loss
                # ------------------------------------------------

                loss = F.mse_loss(
                    current_q,
                    target_q
                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                losses.append(
                    float(loss.item())
                )

        # --------------------------------------------------------
        # Epsilon decay
        # --------------------------------------------------------

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.decay
        )

        # --------------------------------------------------------
        # Synchronize target network
        # --------------------------------------------------------

        self.target.load_state_dict(
            self.net.state_dict()
        )

        return total_reward, losses
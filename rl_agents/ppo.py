import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base_agent import BaseAgent


class ActorCritic(nn.Module):
    def __init__(self, n_state, n_actions):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)

        logits = self.actor(features)
        value = self.critic(features)

        return logits, value


class PPOAgent(BaseAgent):

    def __init__(
        self,
        n_state,
        n_actions,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        epochs=4,
        entropy_coef=0.01
    ):
        self.n_state = n_state
        self.n_actions = n_actions

        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = ActorCritic(
            n_state,
            n_actions
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

    def act(self, state, training=True):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(state_tensor)

        distribution = Categorical(logits=logits)

        if training:
            action = distribution.sample()
        else:
            action = torch.argmax(logits, dim=1)

        return int(action.item())

    def train_episode(self, env):

        state, _ = env.reset()

        states = []
        actions = []
        rewards = []
        old_log_probs = []
        values = []

        done = False

        # --------------------------------------------------
        # Collect trajectory
        # --------------------------------------------------
        while not done:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                logits, value = self.model(state_tensor)

                distribution = Categorical(
                    logits=logits
                )

                action = distribution.sample()
                log_prob = distribution.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(
                int(action.item())
            )

            done = terminated or truncated

            states.append(state)
            actions.append(int(action.item()))
            rewards.append(float(reward))
            old_log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))

            state = next_state

        # --------------------------------------------------
        # Calculate discounted returns
        # --------------------------------------------------
        returns = []

        discounted = 0.0

        for reward in reversed(rewards):

            discounted = (
                reward
                + self.gamma * discounted
            )

            returns.insert(0, discounted)

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )

        states_tensor = torch.tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device
        )

        actions_tensor = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        old_log_probs_tensor = torch.tensor(
            old_log_probs,
            dtype=torch.float32,
            device=self.device
        )

        values_tensor = torch.tensor(
            values,
            dtype=torch.float32,
            device=self.device
        )

        # --------------------------------------------------
        # Advantage calculation
        # --------------------------------------------------
        advantages = returns - values_tensor

        # Normalize advantages
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        # --------------------------------------------------
        # PPO optimization
        # --------------------------------------------------
        for _ in range(self.epochs):

            logits, new_values = self.model(
                states_tensor
            )

            distribution = Categorical(
                logits=logits
            )

            new_log_probs = distribution.log_prob(
                actions_tensor
            )

            entropy = distribution.entropy().mean()

            # Probability ratio
            ratios = torch.exp(
                new_log_probs
                - old_log_probs_tensor
            )

            # Unclipped objective
            objective1 = (
                ratios * advantages
            )

            # Clipped objective
            clipped_ratios = torch.clamp(
                ratios,
                1.0 - self.clip,
                1.0 + self.clip
            )

            objective2 = (
                clipped_ratios * advantages
            )

            # PPO actor loss
            actor_loss = -torch.min(
                objective1,
                objective2
            ).mean()

            # Critic loss
            critic_loss = nn.functional.mse_loss(
                new_values.squeeze(1),
                returns
            )

            # Total loss
            loss = (
                actor_loss
                + 0.5 * critic_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                0.5
            )

            self.optimizer.step()

        return float(sum(rewards))
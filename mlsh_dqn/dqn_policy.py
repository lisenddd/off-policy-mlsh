import tianshou as ts
import numpy as np
import torch
import torch.nn as nn
import wandb

class DQNPolicy:
    """
    DQN policy that gives discrete output
    """
    def __init__(self, input_size, output_size, lr, batch_size):
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Net(input_size, output_size, self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.policy = ts.policy.DQNPolicy(model, self.optimizer, estimation_step=3, target_update_freq=400)
        self.memory = ts.data.ReplayBuffer(size=20000)
        self.steps = 0
        self.train_steps = 0
        self.start_eps = 0.5
        self.policy.set_eps(self.start_eps)
        self.batch_size = batch_size

    def act(self, batch, identifier):
        action = self.policy(batch).act
        self.steps += 1
        if self.steps % 5 == 0:
            # train every 10 steps
            self.steps = 0
            self._train(identifier)
        return action

    def _train(self, identifier):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        # assert batch[0].obs.shape[0] == self.batch_size
        batch = self.policy.process_fn(batch[0], self.memory, batch[1])
        loss = self.policy.learn(batch[0])
        wandb.log({identifier + "loss": loss})
        self.train_steps += 1
        self.eps_annealing()

    def store(self, obs, act, rew, done, obs_next):
        self.memory.add(obs, act, rew, done, obs_next)

    def eps_annealing(self):
        if self.train_steps <= 50000:
            self.policy.set_eps(self.start_eps)
        elif self.train_steps <= 100000:
            eps = self.start_eps - (self.train_steps - 50000) / 90000 * (0.9 * self.start_eps)
            self.policy.set_eps(eps)
        else:
            self.policy.set_eps(0.05)


class QNetwork(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device

        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(128, 128)
        self.fc_q = nn.Linear(128, 128)

        self.value = nn.Linear(128, 1)
        self.q = nn.Linear(128, np.prod(action_shape))

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        obs.to(self.device)
        y = self.relu(self.fc1(obs))
        value = self.relu(self.fc_value(y))
        q = self.relu(self.fc_q(y))

        value = self.value(value)
        q = self.q(q)

        logits = q - q.mean(dim=1, keepdim=True) + value

        return logits, state


class Net(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        obs.to(self.device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state
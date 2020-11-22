import tianshou as ts
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tianshou.utils.net.common import Net

class DQNPolicy:
    """
    DQN policy that gives discrete output
    """
    def __init__(self, input_size, output_size, lr, batch_size, dueling=False, per=False, n_step=3):
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dueling:
            model = Net(1, input_size, output_size, device=self.device, dueling=(1, 1)).to(self.device)
        else:
            model = Net(2, input_size, output_size, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.policy = ts.policy.DQNPolicy(model, self.optimizer, estimation_step=n_step, target_update_freq=400)

        if not per:
            self.memory = ts.data.ReplayBuffer(size=15000)
        else:
            self.memory = ts.data.PrioritizedReplayBuffer(size=15000, alpha=0.6, beta=0.4)
        self.per = per

        self.train_steps = 0
        self.start_eps = 0.5
        # self.start_beta = 0.4
        self.policy.set_eps(self.start_eps)
        self.batch_size = batch_size

    def act(self, batch, identifier, train):
        result = self.policy(batch)
        if train:
            self._train(identifier)
            self.train_steps += 1
        return result.act[0]

    def _train(self, identifier):
        if len(self.memory) < self.batch_size:
            return
        # batch, indice = self.memory.sample(self.batch_size)
        # assert batch[0].obs.shape[0] == self.batch_size
        # batch = self.policy.process_fn(batch, self.memory, indice)
        # loss = self.policy.learn(batch)
        # self.policy.post_process_fn(batch, self.memory, indice)
        loss = self.policy.update(self.batch_size, self.memory)
        self.eps_annealing()

        wandb.log({"low_" + identifier + "_loss": loss})
        wandb.log({"low_" + identifier + "_eps": self.policy.eps})
        # if self.per:
        #     wandb.log({"low_" + identifier + "_beta": self.memory._beta})

    def store(self, obs, act, rew, done, obs_next):
        self.memory.add(obs, act, rew, done, obs_next)

    def eps_annealing(self):
        if self.train_steps <= 10000:
            self.policy.set_eps(self.start_eps)
        elif self.train_steps <= 50000:
            eps = self.start_eps - (self.train_steps - 10000) / 40000 * (0.9 * self.start_eps)
            self.policy.set_eps(eps)
        else:
            self.policy.set_eps(0.05)
        #
        # if self.per:
        #     if self.train_steps <= 20000:
        #         self.memory._beta = self.start_beta
        #     elif self.train_steps <= 100000:
        #         beta = self.start_beta + (self.train_steps - 20000) / 80000 * (0.9 * self.start_beta)
        #         self.memory._beta = beta
        #     else:
        #         self.memory._beta = 1


class DuelingNetwork(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device

        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_q = nn.Linear(128, 128)

        self.value = nn.Linear(128, 1)
        self.q = nn.Linear(128, np.prod(action_shape))

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        obs.to(self.device)
        y = F.relu(self.fc1(obs))
        value = F.relu(self.fc_value(y))
        q = F.relu(self.fc_q(y))

        value = self.value(value)
        q = self.q(q)

        logits = q - q.mean(dim=1, keepdim=True) + value

        return logits, state


class MyNet(nn.Module):
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


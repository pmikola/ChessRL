import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from numba import cuda, vectorize, guvectorize, jit, njit
from torch.autograd import Variable

device = torch.device("cuda")


class Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.num_of_lstm_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=self.num_of_lstm_layers, dropout=0.05,
        #                      batch_first=True)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.out = nn.Softmax(dim=0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1 / np.sqrt(self.input_size))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        # TODO make lstm work with code
        a = F.relu(self.linear1(x))
        b = F.relu(self.linear2(a))
        c = self.linear3(F.dropout(b, 0.1))
        out = self.out(c)
        return out

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        else:
            pass
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Qtrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.loss = None
        self.acc = None

    @staticmethod
    # @njit(nopython=True, parallel=True)
    def Q_fun(model, state, game_over, reward, gamma, next_state, action):
        # Predicted quality function of the given state + greedy
        # Qnew(state,action) = (1 - alpha) * Qold(state,action) + alpha(rewards + gamma*max(state',action')

        Qprediction = model(state)
        Qtarget = Qprediction.clone()
        for index in range(len(game_over)):
            Q_new = reward[index]
            if not game_over[index]:
                # Temporal Difrence
                Q_new = reward[index] + gamma * torch.max(model(next_state[index]))  # The Q learning iter
            # Update Qvalue as in teh Bellman equation
            Qtarget[index][torch.argmax(action).item()] = Q_new

        # 2 :Q_new =  r + gamma *max(next_predicted Q value)
        # prediction.clone()
        # prediction[torch.argmax(action)] = Q_new
        return Qtarget, Qprediction

    def train_step(self, state_old, action, reward, state_new, done):
        # print(state_old[0])

        state = torch.flatten(torch.tensor(np.concatenate((state_old[0], state_old[1])), dtype=torch.float))
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        next_state = torch.flatten(torch.tensor(np.concatenate((state_new[0], state_new[1])), dtype=torch.float))
        # (n,x)

        if len(state.shape) == 1:
            # state has one dim
            # (1,x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)  # tuple with only one value

        target, prediction = Qtrainer.Q_fun(self.model, state, done, reward, self.gamma, next_state, action)

        self.optim.zero_grad()
        loss = self.criterion(target, prediction)
        self.loss = loss.item()
        loss.backward()
        self.optim.step()

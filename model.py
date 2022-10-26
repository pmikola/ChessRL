import random
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
        # self.out = nn.Softmax(dim=0)

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1 / np.sqrt(self.input_size))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input):
        # TODO make lstm work with code
        a = F.relu(self.linear1(input))
        b = F.relu(self.linear2(F.dropout(a, 0.1)))
        out = self.linear3(b)
        # out = self.out(c)
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
    def __init__(self, model, lr, gamma, alpha):
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss = None
        self.acc = None

    @staticmethod
    # @njit(nopython=True, parallel=True)
    def Q_fun(model, state, game_over, reward, gamma, alpha, next_state, action):
        # RL Double-Q Learning Algorithm
        # TODO: MAKE IT FROM CPU TO GPU !!!!!!!!!!!!!!
        for games_n in range(0, len(game_over)):
            if len(game_over) > 1:
                indices_model = torch.tensor([games_n])

                state = torch.index_select(state, 0, indices_model)
                next_state = torch.index_select(next_state, 0, indices_model)
                reward = reward[games_n]
                action = action[games_n]
                Qacurr = model(state[0])
                Qbcurr = model(state[0])
                Qaestim = Qacurr.clone()
                Qbestim = Qbcurr.clone()
                Qanext = model(next_state[0])
                Qbnext = model(next_state[0])
                # print(Qbnext.size())
                # time.sleep(3)
            else:
                Qacurr = model(state)
                Qbcurr = model(state)
                Qaestim = Qacurr.clone()
                Qbestim = Qbcurr.clone()
                Qanext = model(next_state)
                Qbnext = model(next_state)

            # print(range(len(game_over)))
            random_choice = random.randint(0, 1)
            if games_n == 0:
                random_choice = 0
            elif games_n == 1:
                random_choice = 1
            else:
                pass
            # random_choice = 0
            if random_choice == 0:
                #a = torch.argmax(Qanext)  # sample action a
                indices_a = torch.tensor([action.type(torch.int64)])
                Qbnext = torch.max(torch.index_select(Qbnext, 0, indices_a))
                Qaestim += alpha * (reward + gamma * Qbnext - Qacurr)
            else:
                #b = torch.argmax(Qbnext)  # sample action b
                indices_b = torch.tensor([action.type(torch.int64)])
                Qanext = torch.max(torch.index_select(Qanext, 0, indices_b))
                Qbestim += alpha * (reward + gamma * Qanext - Qbcurr)

            # Mean in estim between two Q value estimators and Q current
            Qab_estim = torch.div(torch.add(Qaestim, Qbestim), 2.)
            Qab_curr = torch.div(torch.add(Qacurr, Qbcurr), 2.)

            return Qab_curr, Qab_estim

    def train_step(self, state_old, action, reward, state_new, done):
        set_flag = 0
        if type(action) is tuple:
            action = torch.from_numpy(np.array(action, dtype=np.float32))
            state = torch.flatten(torch.from_numpy(np.array(state_old, dtype=np.float32)), start_dim=1)
            reward = torch.from_numpy(np.array(reward, dtype=np.float32))
            next_state = torch.flatten(torch.from_numpy(np.array(state_new, dtype=np.float32)), start_dim=1)
            set_flag = 1
            # print(state.size(), next_state.size(), action.size(), reward.size())
            # done = (done,)  # tuple with only one value\
            # print(done)
            Qnetwork, Q_target = Qtrainer.Q_fun(self.model, state, done, reward, self.gamma, self.alpha, next_state,
                                                action)

            self.optim.zero_grad()
            loss = self.criterion(Qnetwork, Q_target)
            self.loss = loss.item()
            loss.backward()
            self.optim.step()
        if set_flag == 1:
            pass
        else:
            action = torch.from_numpy(np.asarray(action, dtype=np.float32))
            state = torch.flatten(torch.from_numpy(state_old))
            reward = torch.tensor(np.array(reward), dtype=torch.float)
            next_state = torch.flatten(torch.from_numpy(state_new))
            done = (done,)  # tuple with only one value\

            Qnetwork, Q_target = Qtrainer.Q_fun(self.model, state, done, reward, self.gamma, self.alpha, next_state,
                                                action)

            self.optim.zero_grad()
            loss = self.criterion(Qnetwork, Q_target)
            self.loss = loss.item()
            loss.backward()
            self.optim.step()

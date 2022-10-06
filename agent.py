import sys
import threading
import time

import chess
import torch
import random
import numpy as np
from collections import deque  # datastructure to store memory vals

from PyQt5.QtWidgets import QApplication

from model import Qnet, Qtrainer
from disp_progress import plot
from ChessGame import ChessGameRL
from fen_to_board import fenToBoard

########################### BELLMAN EQUATiON ##############################
# NewQ(state,action) = Q(state,action) + lr[R(state,action) + gamma*maxQ'(state',action') - Q(state,action(]
# NewQ(state,action) - New Q vale for that state and action
# Q(state,action) - current Q value in state and action
# lr - learning rate
# R(state,action) - reward for thaking that action and that state
# gamma- discount rate
# maxQ'(state',action') - Maximum expected future reward for given new state and all possible actions at that new state
########################### BELLMAN EQUATiON ##############################

MAX_MEMORY = 100_000
BATCH_SIZE = 16
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        # self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # self.model = Qnet(5248, 10000, 5184)
        # self.trainer = Qtrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        legals = list(game.chessboard.legal_moves)
        print(legals)
        chessboard_state = fenToBoard(game)
        moves = np.zeros(5184, dtype=float)
        for m in legals:
            str_choosen_move = chess.Move.uci(m)
            legals_index = np.where(game.all_moves == str_choosen_move)[0]
            moves[legals_index] = 1.
        legal_moves = np.reshape(moves, (648, 8))
        #state = np.concatenate((chessboard_state, legal_moves))
        state = [
            chessboard_state,
            legal_moves
        ]
        return np.array(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        legals = list(game.chessboard.legal_moves)
        print(state)
        # random moves: tradeoff exploration / exploitation
        final_move = np.zeros(5184, dtype=float)
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            random_move = np.random.randint(0, len(legals))
            choosen_move = legals[random_move]
            str_choosen_move = chess.Move.uci(choosen_move)
            final_move = np.where(game.all_moves == str_choosen_move)[0]
            #final_move[choosen_index] = 1.
        else:
            game_state = np.concatenate((state[0],state[1]))
            print(game_state)
            game_state_tensor = torch.tensor(game_state, dtype=torch.float)
            time.sleep(2)
            prediction = self.model(game_state_tensor)
            final_move = torch.argmax(prediction).item()
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    iterator = 0
    MainThread = QApplication(sys.argv)
    agent = Agent()
    game = ChessGameRL(chess.Board())
    thread = threading.Thread(target=game_loop(game, agent, plot_scores, plot_mean_scores, total_score, record))
    thread.start()
    game.show()
    MainThread.exec()
    MainThread.processEvents()
    del game, thread


def game_loop(game, agent, plot_scores, plot_mean_scores, total_score, record):
    running = True
    while running:
        np.random.seed(2022)

        # get old state
        state_old = agent.get_state(game)

        final_move_index = agent.get_action(state_old, game)
        # perform move and get new state
        #p = chess.Move.from_uci(str(final_move))
        p = game.all_moves[final_move_index]
        game.chessboard.push(p)

        reward, done, score = game.play_step(final_move)
        game.set_chessboard()
        time.sleep(0.035)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()

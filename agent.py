import sys
import threading
import time

import chess
import matplotlib
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
BATCH_SIZE = 32
LR = 0.001
white = True
black = False


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Qnet(5248, 1000, 5184)
        self.trainer = Qtrainer(self.model, lr=LR, gamma=self.gamma)
        self.game = None
        self.agent = None
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0.
        self.record = 0.
        self.reward = 0.

    def get_state(self, game):
        legals = list(game.chessboard.legal_moves)
        chessboard_state = fenToBoard(game)
        moves = np.zeros(5184, dtype=float)
        for m in legals:
            str_choosen_move = chess.Move.uci(m)
            legals_index = np.where(np.array(game.all_moves) == str_choosen_move)[0]
            moves[legals_index] = 1.
        legal_moves = np.reshape(moves, (648, 8))
        # state = np.concatenate((chessboard_state, legal_moves))
        state = [
            chessboard_state,
            legal_moves
        ]
        # print(game.chessboard.legal_moves.count())
        return np.array(state, dtype=object)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        # time.sleep(10)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        print(actions[0])
        print(np.array(states).shape)
        self.trainer.train_step(states[0], actions[0], rewards[0], next_states[0], dones[0])
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game, reward):
        legals = list(game.chessboard.legal_moves)
        # random moves: tradeoff exploration / exploitation
        # final_move = np.zeros(5184, dtype=float)
        self.epsilon = 80 - self.n_games
        random_move = np.random.randint(0, len(legals))
        choosen_move = legals[random_move]
        str_choosen_move = chess.Move.uci(choosen_move)
        promotion = None
        try:
            if 'q' or 'r' or 'b' or 'n' in str_choosen_move[4]:
                promotion = str_choosen_move[4]
                str_choosen_move = str_choosen_move[0:4]
                print('PROMOOOOOOOOOOOOOOOO!!!!!')
                reward += 3
        except:
            pass

        final_move = np.where(np.array(game.all_moves) == str_choosen_move)[0]
        rand_move = final_move
        if random.randint(0, 200) < self.epsilon:
            pass
        else:
            game_state = np.concatenate((state[0], state[1]))
            # print(game_state)
            game_state_tensor = torch.tensor(game_state, dtype=torch.float)
            prediction = self.model(torch.flatten(game_state_tensor))
            final_move = torch.argmax(prediction).item()
        # print(final_move, rand_move, str_choosen_move)

        return final_move, rand_move, reward, promotion


def train(agent):
    for i in range(0, 2):
        MainThread = QApplication(sys.argv)
        agent.game = ChessGameRL(chess.Board())
        thread = threading.Thread(target=game_loop)
        thread.start()
        agent.game.show()
        MainThread.exec()
        MainThread.processEvents()
        del agent.game, thread


def game_loop():
    np.random.seed(4)
    running = True
    color = white
    while running:

        # get old state
        state_old = agent.get_state(agent.game)
        # print(state_old[0])
        # time.sleep(5)
        final_move_index, random_legal_move_index, reward, promotion = agent.get_action(state_old, agent.game,
                                                                                        agent.reward)
        # perform move and get new state
        p, agent.reward = agent.game.is_valid_move(agent.game, state_old, final_move_index, random_legal_move_index,
                                                   agent.reward)
        agent.game.chessboard.push(p)
        agent.game.is_promoted(agent.game, p, promotion, color)
        agent.game.set_chessboard()
        # checks
        agent.reward, agent.done, agent.score = agent.game.play_step(agent.game, agent.reward, p)
        # print(reward)
        time.sleep(0.03)
        state_new = agent.get_state(agent.game)
        # train short memory
        agent.train_short_memory(state_old, final_move_index, reward, state_new, agent.done)

        # remember
        agent.remember(state_old, final_move_index, agent.reward, state_new, agent.done)
        agent.reward -= 1.
        if color == white:
            color = black
        else:
            color = white
        if agent.done:
            # train long memory, plot result
            print("THE GAME IS OVER AND YOU NEED TO HANDLE THIS NOW!")
            agent.n_games += 1
            agent.train_long_memory()

            if agent.score > agent.record:
                agent.record = agent.score
                agent.model.save()

            print('Game', agent.n_games, 'Score', agent.score, 'Record:', agent.record)

            agent.plot_scores.append(agent.score)
            agent.total_score += agent.score
            mean_score = agent.total_score / agent.n_games
            agent.plot_mean_scores.append(mean_score)
            plot(agent.plot_scores, agent.plot_mean_scores)
            # RESET GAME
            agent.done = False


if __name__ == '__main__':
    agent = Agent()
    train(agent)

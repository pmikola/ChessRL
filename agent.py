import sys
import threading
import time
from IPython import display
import chess
import matplotlib
import torch
import random
import numpy as np
from collections import deque  # datastructure to store memory vals
import chess.engine
from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt
from stockfish import Stockfish
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
BATCH_SIZE = 128
LR = 0.001
white = True
black = False
matplotlib.use('Qt5Agg')


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 5  # randomness
        self.gamma = 0.9  # discount rate
        self.alpha = 0.2  # learning rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Qnet(5248, 5000, 5184, 2)
        self.trainer = Qtrainer(self.model, lr=0.001, gamma=self.gamma, alpha=self.alpha)
        self.game = None
        self.StockFish_Enginge = None
        self.agent = None
        self.plot_scores = []
        self.plot_mean_scores = []
        self.plot_loss = []
        self.total_score = 0.
        self.record = -1000.
        self.reward = 0.
        self.agent = []
        self.plot_n_games = []
        self.done = False

    def get_state(self, game):
        legals = list(game.chessboard.legal_moves)
        chessboard_state = fenToBoard(game)
        moves = np.zeros(5184, dtype=float)
        for m in legals:
            str_choosen_move = chess.Move.uci(m)
            legals_index = np.where(np.array(game.all_moves) == str_choosen_move)[0]
            moves[legals_index] = 1.
        legal_moves = np.reshape(moves, (648, 8))
        state = np.concatenate((np.array(chessboard_state, dtype=np.float32), np.array(legal_moves, dtype=np.float32)),
                               axis=0)
        return state

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # print(dones[0])

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):

        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game, reward, color):
        legals = list(game.chessboard.legal_moves)
        # random moves: tradeoff exploration / exploitation
        # EPSILON GREEDY ALGORITHM
        self.epsilon = 80 - self.n_games
        random_move = np.random.randint(0, len(legals))
        choosen_move = legals[random_move]
        str_choosen_move = chess.Move.uci(choosen_move)
        promotion = None
        try:
            if 'q' or 'r' or 'b' or 'n' in str_choosen_move[4]:
                promotion = str_choosen_move[4]
                print(promotion)
                str_choosen_move = str_choosen_move[0:4]
                reward += 5
        except:
            pass
        action = int(np.where(np.array(game.all_moves) == str_choosen_move)[0])
        rand_action = action

        if random.randint(0, 200) < self.epsilon:
            pass
        else:
            game_state_tensor = torch.from_numpy(state)
            prediction = self.model(torch.flatten(game_state_tensor))
            action = torch.argmax(prediction).item()

        return action, rand_action, reward, promotion


def train(agent):
    MainThread = QApplication(sys.argv)
    agent.game = ChessGameRL(chess.Board())

    agent.StockFish_Enginge = chess.engine.SimpleEngine.popen_uci(
        r"C:\PYTHON_PROJECTS\ChessRL\stockfish_15_win_x64_avx2\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe")

    thread = threading.Thread(target=game_loop)
    thread.start()
    agent.game.show()
    MainThread.exec()
    MainThread.processEvents()
    del agent.game, thread


def game_loop():
    np.random.seed(7)
    running = True
    agent.done = False
    color = white
    start = time.time()
    while running:
        try:
            if color:
                # get old state
                state_old = agent.get_state(agent.game)

                action, random_action, reward, promotion = agent.get_action(state_old,
                                                                            agent.game,
                                                                            agent.reward, color)
                # perform move and get new state
                p, agent.reward = agent.game.is_valid_move(agent.game, state_old, action,
                                                           random_action,
                                                           agent.reward)
                if p is None:
                    print(p)
                    agent.reward -= 20.
                    agent.reward, agent.done, agent.score = agent.game.play_step(agent.game, agent.reward, p, color)
                    agent.train_short_memory(state_old, action, reward, state_old, agent.done)
                    agent.remember(state_old, action, agent.reward, state_old, agent.done)
                    if color == white:
                        color = black
                    else:
                        color = white
                else:
                    agent.game.chessboard.push(p)
                    agent.game.is_promoted(agent.game, p, promotion, color)
                    agent.game.set_chessboard()


                    # checks
                    agent.reward, agent.done, agent.score = agent.game.play_step(agent.game, agent.reward, p, color)

                    state_new = agent.get_state(agent.game)

                    reward = ChessGameRL.CapturedPieceCheck(agent.game, state_old, state_new, p, reward,
                                                            color)

                    # train short memory
                    # TD(0) Learning

                    agent.train_short_memory(state_old, action, reward, state_new, agent.done)

                    # remember

                    agent.remember(state_old, action, agent.reward, state_new, agent.done)

                    agent.reward -= 1.
                    if color == white:
                        color = black
                    else:
                        color = white
                time.sleep(0.025)
            else:
                result = agent.StockFish_Enginge.play(agent.game.chessboard, chess.engine.Limit(time=0.050))
                agent.game.chessboard.push(result.move)
                agent.game.set_chessboard()
                agent.reward, agent.done, agent.score = agent.game.play_step(agent.game, agent.reward, result.move,
                                                                             color)
                if color == white:
                    color = black
                else:
                    color = white
                time.sleep(0.025)

            # time.sleep(1)

            if agent.done:
                # train long memory, plot result
                # TD(N) (aka. MonteCarlo like) Learning but from past experience
                agent.n_games += 1
                agent.train_long_memory()

                if agent.score > agent.record:
                    agent.record = agent.score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', agent.score, 'Record:', agent.record, 'Loss: ',
                      agent.trainer.loss)

                agent.plot_scores.append(agent.score)
                agent.total_score += agent.score
                agent.mean_score = agent.total_score / agent.n_games
                agent.plot_mean_scores.append(agent.mean_score)
                agent.plot_loss.append(agent.trainer.loss)
                agent.plot_n_games.append(agent.n_games)
                ##plot(agent.plot_scores, agent.mean_score, agent.plot_loss)

                # RESET GAME
                agent.done = False
                running = False
                agent.score = 0.
                agent.reward = 0.
                agent.game.close()
                agent.StockFish_Enginge.quit()
                end = time.time()
                print("Time: ", end - start, "[ s ]")
        # except:
        except Exception as e:
            print(e)
            agent.done = True
            agent.reward = 0


if __name__ == '__main__':
    agent = Agent()
    while agent.total_score <= 10000:
        train(agent)
    print('Woow - Thats awesome !!')

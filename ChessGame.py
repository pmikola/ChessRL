import time

import chess
import chess.svg
from IPython.core.display import SVG
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
import numpy as np
import sys
import threading


class ChessGameRL(QWidget):
    def __init__(self, board):
        super().__init__()
        self.setGeometry(100, 100, 1100, 1100)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 1080, 1080)
        self.chessboard = board
        self.all_moves = ChessGameRL.generate_all_moves()

    def set_chessboard(self):
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    @staticmethod
    def generate_all_moves():
        all_moves = np.array(np.zeros(5184), dtype=str)  # with queens
        middle_chars_arr = np.array(np.zeros(72), dtype=str)
        k = 0
        for a in 'abcdefghq':  # + 'q' if queen transformation at the other end of the chessboard
            for b in '12345678':
                middle_chars_arr[k] = a + b
                k += 1
        k = 0
        for c in middle_chars_arr:
            for d in middle_chars_arr:
                all_moves[k] = c + d
                k += 1
        return all_moves

    @staticmethod
    def is_valid_move(game, state, final_move_index, random_legal_move_index, reward):
        move = None
        exist_in_legals = np.ndarray.flatten(state[1])[final_move_index]
        if final_move_index is None:
            pass
        else:
            if exist_in_legals != 0:
                move = game.all_moves[final_move_index]
        if random_legal_move_index != 0:
            if exist_in_legals == 0:
                move = game.all_moves[random_legal_move_index]
                reward += -10
            else:
                pass

        print(final_move_index, random_legal_move_index)  # TODO if one of those
        # TODO values are empty its mean that there are no legal moves to do
        # TODO need to handle that by passing turn to the opponent

        p = chess.Move.from_uci(str(move[0]))

        return p, reward

    @staticmethod
    def play_step(game, reward, move_idx):
        done = game.chessboard.is_checkmate()
        if game.chessboard.is_variant_win() is True:
            reward += 10
            done = True
        elif game.chessboard.is_variant_loss() is True:
            done = True
            reward -= 10
        elif game.chessboard.is_variant_draw() is True:
            done = True
        elif game.chessboard.is_stalemate() is True:
            done = True

        reward = ChessGameRL.CapturedPiece(game, move_idx, reward)
        score = reward  # temporarily for development purposes
        return reward, done, score

    @staticmethod
    def CapturedPiece(game, move, reward):
        if game.chessboard.is_capture(move):
            reward += 1
            if game.chessboard.is_en_passant(move):
                reward += 1
            else:
                reward += 0
        else:
            reward += 0
        # print(move)
        return reward
# def single_game():
#     MainThread = QApplication(sys.argv)
#     global game
#     game = ChessGameRL(chess.Board())
#     thread = threading.Thread(target=game_loop)
#     thread.start()
#     game.show()
#     MainThread.exec()
#     MainThread.processEvents()
#     del game, thread
#
#
# def game_loop():
#     running = True
#     iterator = 0
#     # np.random.seed(2022)
#     while running:
#         # legals = game.chessboard.legal_moves
#         llegals = list(game.chessboard.legal_moves)
#         random_move = np.random.randint(0, len(llegals))
#         chosen_move = llegals[random_move]
#         # print(chosen_move)
#         # print(legals.count())
#         p = chess.Move.from_uci(str(chosen_move))
#         game.chessboard.push(p)
#         game.set_chessboard()
#         time.sleep(0.035)
#         iterator += 1
#         if iterator >= 10:
#             running = False
#             game.close()
#
#
#
# for i in range(0, 5):
#     single_game()

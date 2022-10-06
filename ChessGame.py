import time

import chess
import chess.svg
from IPython.core.display import SVG
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
import numpy as np
import sys
import threading


##################### CLASSES #####################


class ChessGameRL(QWidget):
    def __init__(self, board):
        super().__init__()
        self.setGeometry(100, 100, 1100, 1100)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 1080, 1080)
        self.chessboard = board
        self.all_moves = generate_all_moves()

    def set_chessboard(self):
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)


#################### CLASSES #####################
###################### FUNS ######################
def generate_all_moves():
    all_moves = np.array(np.zeros(5184), dtype=str) # with queens
    middle_chars_arr = np.array(np.zeros(72), dtype=str)
    k = 0
    for a in 'abcdefghq':
        for b in '12345678':
            middle_chars_arr[k] = a + b
            k += 1
    k = 0
    for c in middle_chars_arr:
        for d in middle_chars_arr:
            all_moves[k] = c + d
            k += 1
    return all_moves

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

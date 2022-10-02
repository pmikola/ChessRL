import time

import chess
import chess.svg
from IPython.core.display import SVG
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
import numpy as np


####################### INIT ######################
class MainGame(QWidget):
    def __init__(self, board):
        super().__init__()
        self.setGeometry(100, 100, 1100, 1100)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 1080, 1080)
        self.chessboard = board

    def set_chessboard(self):
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)


####################### INIT ######################
for i in range(0, 5):
    np.random.seed(2022)
    app = QApplication([])
    game_var = "game"
    if game_var in globals() or game_var in locals():
        print("ok")
    else:
        game = MainGame(chess.Board())
    legals = game.chessboard.legal_moves
    llegals = list(game.chessboard.legal_moves)
    # for i in range(0,len(llegals)):
    #     print(llegals[i])
    # np.random.seed(2022)
    random_move = np.random.randint(0, len(llegals))
    chosen_move = llegals[random_move]
    print(chosen_move)
    print(legals.count())
    p = chess.Move.from_uci(str(chosen_move))
    game.chessboard.push(p)
    game.set_chessboard()
    game.show()
    time.sleep(1)
    app.exec()

    del app, game

# print(board.legal_moves)

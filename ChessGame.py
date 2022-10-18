import time

import chess
import chess.svg
from IPython.core.display import SVG
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
import numpy as np
import sys
import threading
from numba import cuda, vectorize, guvectorize, jit, njit


class ChessGameRL(QWidget):
    def __init__(self, board):
        super().__init__()
        self.setGeometry(100, 100, 700, 700)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 690, 690)
        self.chessboard = board
        self.all_moves = ChessGameRL.generate_all_moves()
        self.pieces_counter = 260

    def set_chessboard(self):
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    @staticmethod
    def generate_all_moves():
        all_moves = np.array(np.zeros(5184), dtype=str)  # with queens
        middle_chars_arr = np.array(np.zeros(72), dtype=str)
        k = 0
        for a in 'abcdefgh':  # + 'q' if queen transformation at the other end of the chessboard
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
        exist_in_legals = np.ndarray.flatten(state[1])[final_move_index]
        if exist_in_legals.any():
            move = game.all_moves[final_move_index]
            reward += 25
        else:
            move = game.all_moves[random_legal_move_index]
            reward -= 25
        if random_legal_move_index is None and final_move_index is None:
            reward -= 10
            return None, reward
        else:
            str_move = chess.Move.from_uci(str(move))
            return str_move, reward

    @staticmethod
    def is_promoted(game, p, promotion, color):
        if promotion is None:
            pass
        else:
            move_str = chess.Move.uci(p)[2:4]
            if color:
                game.chessboard.set_piece_at(chess.parse_square(move_str), chess.Piece.from_symbol(promotion.upper()),
                                             promoted=False)
            else:
                game.chessboard.set_piece_at(chess.parse_square(move_str), chess.Piece.from_symbol(promotion),
                                             promoted=False)
            # print(move_str)
            # time.sleep(2)

    @staticmethod
    def play_step(game, reward, move_idx, color):
        done = False

        if color is False:
            if game.chessboard.is_checkmate() and done is False:
                print('Chekmate Black Win')
                reward -= 100
                done = True
            if game.chessboard.is_variant_win() is True and done is False:
                print('White Win!')
                reward += 100
                done = True
            elif game.chessboard.is_variant_loss() is True and done is False:
                print('Black Win!')
                done = True
                reward -= 100
            elif game.chessboard.is_variant_draw() is True and done is False:
                print('It is a DRAW!')
                done = True
            elif game.chessboard.is_stalemate() is True and done is False:
                done = True
            elif game.chessboard.is_fifty_moves() is True and done is False:
                print('50 moves passed :(!')
                done = True
                reward -= 100
            elif game.chessboard.is_fivefold_repetition() is True and done is False:
                done = True
                reward -= 50
            elif game.chessboard.is_game_over() is True and done is False:
                print('GAME OVER ! ')
                done = True
        else:
            if game.chessboard.is_checkmate() and done is False:
                print('Chekmate White Win')
                reward += 100
                done = True
            if game.chessboard.is_variant_win() is True and done is False:
                print('White Win!')
                reward -= 100
                done = True
            elif game.chessboard.is_variant_loss() is True and done is False:
                print('Black Win!')
                done = True
                reward += 100
            elif game.chessboard.is_variant_draw() is True and done is False:
                print('It is a DRAW!')
                done = True
            elif game.chessboard.is_stalemate() is True and done is False:
                done = True
            elif game.chessboard.is_fifty_moves() is True and done is False:
                print('50 moves passed :(!')
                done = True
                reward -= 100
            elif game.chessboard.is_fivefold_repetition() is True and done is False:
                done = True
                reward -= 50
            elif game.chessboard.is_game_over() is True and done is False:
                print('GAME OVER ! ')
                done = True

        score = reward  # temporarily for development purposes
        return reward, done, score

    @staticmethod
    def CapturedPieceCheck(game, state_old, state_new, move, reward, color):
        # print('old : ', np.sum(state_old[0]))
        # print('new : ', np.sum(state_new[0]))
        old = np.sum(state_old[0])
        new = np.sum(state_new[0])
        b_cap = abs(game.pieces_counter - int(new))
        capture = int(np.abs(old - new))
        if color is True:
            if move is None:
                pass
            else:
                if capture == 0 and b_cap == 0:
                    pass
                else:
                    if b_cap >= 11:
                        reward -= 3
                    if 0 < b_cap < 11:
                        reward -= 10
                    game.pieces_counter = int(new)
                if capture >= 11:
                    reward += 3
                if 0 < capture < 11:
                    reward += 10
        return reward

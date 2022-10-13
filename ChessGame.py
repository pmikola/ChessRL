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
        self.setGeometry(100, 100, 700, 700)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 690, 690)
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

            str_move = chess.Move.from_uci(str(move[0]))
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
        if color is True:
            if game.chessboard.is_checkmate():
                print('Chekmate')
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
            if game.chessboard.is_checkmate():
                print('Chekmate')
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

        reward = ChessGameRL.CapturedPiece(game, move_idx, reward, color)
        score = reward  # temporarily for development purposes
        return reward, done, score

    @staticmethod
    def CapturedPiece(game, move, reward, color):
        if color is True:
            if move is None:
                pass
            else:
                if game.chessboard.is_capture(move):
                    reward += 5
                    if game.chessboard.is_en_passant(move):
                        reward -= 2
                    else:
                        reward += 0
                else:
                    reward += 0
        else:
            if move is None:
                pass
            else:
                if game.chessboard.is_capture(move):
                    reward -= 5
                    if game.chessboard.is_en_passant(move):
                        reward += 2
                    else:
                        reward += 0
                else:
                    reward += 0
        # print(move)
        return reward

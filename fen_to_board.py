import numpy as np


def fenToBoard(game):
    chessboard_state = np.zeros((8, 8))
    figures_position = game.chessboard.fen()
    iter_x = 0
    iter_y = 0
    figures_position_split = figures_position.split('/')
    for c in range(0, 8):
        fen_row = figures_position_split[c]
        if len(fen_row) >= 7:
            fen_row = fen_row.split(' ')
        else:
            pass
        for k in fen_row[0]:
            match k:
                case 'r':
                    chessboard_state[iter_x, iter_y] = 1.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'R':
                    chessboard_state[iter_x, iter_y] = 2.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'n':
                    chessboard_state[iter_x, iter_y] = 3.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'N':
                    chessboard_state[iter_x, iter_y] = 4.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'b':
                    chessboard_state[iter_x, iter_y] = 5.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'B':
                    chessboard_state[iter_x, iter_y] = 6.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'q':
                    chessboard_state[iter_x, iter_y] = 7.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'Q':
                    chessboard_state[iter_x, iter_y] = 8.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'k':
                    chessboard_state[iter_x, iter_y] = 9.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'K':
                    chessboard_state[iter_x, iter_y] = 10.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'p':
                    chessboard_state[iter_x, iter_y] = 11.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case 'P':
                    chessboard_state[iter_x, iter_y] = 12.
                    iter_y += 1
                    if iter_y > 7:
                        iter_y = 0
                        iter_x += 1
                case _:
                    if k in '12345678':
                        for m in range(0, int(k) * 1):
                            chessboard_state[iter_x, iter_y] = 0.
                            iter_y += 1
                            if iter_y > 7:
                                iter_y = 0
                                iter_x += 1
                    else:
                        pass
    return chessboard_state

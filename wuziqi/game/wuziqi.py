import game.interfaces
import numpy as np


class WuziqiAction(object):
    def __init__(self, x, y, val):
        self.x = x
        self.y = y
        self.val = val

    def reverse(self):
        return WuziqiAction(self.x, self.y, self.val * -1)


class Position(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class WuziqiGame(game.interfaces.IEnvironment):
    SIDES = [-1, 1]
    WINNING_SIZE = 5

    def __init__(self, board_size):
        self.board_size = board_size
        self.state = np.zeros(board_size)
        self.last_action = WuziqiAction(0, 0, 0)

    def get_state(self):
        return self.state

    def update(self, action):
        self.state[action.x, action.y] = action.val
        self.last_action = action
        return self.state

    def get_available_points(self):
        return np.where(self.state == 0)

    @staticmethod
    def eval_row(row):
        last_val = 0
        repeat_count = 0
        for p in row:
            if p == last_val:
                repeat_count += 1
                if (p in WuziqiGame.SIDES) and repeat_count >= WuziqiGame.WINNING_SIZE:
                    return p
            else:
                last_val = p
                repeat_count = 1
        return 0

    @staticmethod
    def get_diagonal_row(state, start_x, start_y, end_x, end_y, x_step, y_step):
        # print("start_x, end_x, x_step, start_y, end_y, y_step:", start_x, end_x, x_step, start_y, end_y, y_step)
        row = state[range(start_x, end_x, x_step), range(start_y, end_y, y_step)]
        # print(row)
        return row

    @staticmethod
    def get_diagonal_rows(board_size, state):
        rows = []
        for x in range(WuziqiGame.WINNING_SIZE - 1, board_size[0]):
            rows.append(WuziqiGame.get_diagonal_row(state, x, 0, -1, x + 1, -1, 1))
        for y in range(1, board_size[1] - WuziqiGame.WINNING_SIZE + 1):
            rows.append(WuziqiGame.get_diagonal_row(state, board_size[0] - 1, y, y - 1, board_size[1], -1, 1))
        for x in range(board_size[0] - WuziqiGame.WINNING_SIZE, 0, -1):
            rows.append(WuziqiGame.get_diagonal_row(state, x, 0, board_size[0], board_size[1] - x, 1, 1))
        for y in range(board_size[1] - WuziqiGame.WINNING_SIZE + 1):
            rows.append(WuziqiGame.get_diagonal_row(state, 0, y, board_size[0] - y, board_size[1], 1, 1))
        return rows

    @staticmethod
    def eval_state(board_size, state):
        for x in range(board_size[0]):
            val1 = WuziqiGame.eval_row(state[x, :])
            if val1 in WuziqiGame.SIDES:
                return val1
        for y in range(board_size[1]):
            val2 = WuziqiGame.eval_row(state[:, y])
            if val2 in WuziqiGame.SIDES:
                return val2
        for row in WuziqiGame.get_diagonal_rows(board_size, state):
            val3 = WuziqiGame.eval_row(row)
            if val3 in WuziqiGame.SIDES:
                return val3
        return 0

    def is_ended(self):
        return self.eval_state(self.board_size, self.state) != 0 or len(self.get_available_points()[0]) == 0

    def show(self):
        condlist = [self.state == 1, self.state == 0, self.state == -1]
        choicelist = ['X', '-', 'O']
        printable = np.select(condlist, choicelist)

        def print_row(row):
            print(' '.join(row))

        # np.apply_along_axis(print_row, 1, printable)
        for y in range(self.board_size[1]):
            row = []
            for x in range(self.board_size[0]):
                state = self.state[x, y]
                if self.last_action.x == x and self.last_action.y == y:
                    if self.last_action.val == 1:
                        row.append('#')
                    elif self.last_action.val == -1:
                        row.append('%')
                elif state == 1:
                    row.append('X')
                elif state == -1:
                    row.append('O')
                else:
                    row.append('-')
            print(' '.join(row))

        print("#########")

    def clone(self):
        cloned = WuziqiGame(self.board_size)
        cloned.state = self.state.copy()
        return cloned

    def reverse(self):
        cloned = WuziqiGame(self.board_size)
        cloned.state = self.state * -1
        cloned.last_action = self.last_action.reverse()
        return cloned

    def neighbor(self, pos: Position, available_only: bool):
        def get_options(current, boundary):
            if current == 0:
                return [0, 1]
            elif current == (boundary - 1):
                return [-1, 0]
            else:
                return [-1, 0, 1]

        def is_available(x, y):
            return self.state[x, y] == 0

        x_options = get_options(pos.x, self.board_size[0])
        y_options = get_options(pos.y, self.board_size[1])

        positions = []
        for dx in x_options:
            x = pos.x + dx
            for dy in y_options:
                y = pos.y + dy
                if x == pos.x and y == pos.y:
                    continue
                elif available_only:
                    if is_available(x, y):
                        positions.append(Position(x, y))
                else:
                    positions.append(Position(x, y))
        return positions

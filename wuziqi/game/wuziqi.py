import game.interfaces
import numpy as np


class WuziqiAction(object):
    def __init__(self, x, y, val):
        self.x = x
        self.y = y
        self.val = val


class WuziqiGame(game.interfaces.IEnvironment):
    def __init__(self, board_size, winning_size):
        self.board_size = board_size
        self.winning_size = winning_size
        self.state = np.zeros(board_size)
        self.terminal_point_states = [-1, 1]

    def get_state(self):
        return self.state

    def update(self, action):
        self.state[action.x, action.y] = action.val
        return self.state

    def get_available_points(self):
        return np.where(self.state == 0)

    def eval_row(self, row):
        last_val = 0
        repeat_count = 0
        for p in row:
            if p == last_val:
                repeat_count += 1
                if (p in self.terminal_point_states) and repeat_count >= self.winning_size:
                    return p
            else:
                last_val = p
                repeat_count = 0
        return 0

    def get_diagonal_row(self, start_x, start_y, end_x, end_y, x_step, y_step):
        # print("start_x, end_x, x_step, start_y, end_y, y_step:", start_x, end_x, x_step, start_y, end_y, y_step)
        row = self.state[range(start_x, end_x, x_step), range(start_y, end_y, y_step)]
        # print(row)
        return row

    def get_diagonal_rows(self):
        rows = []
        for x in range(self.winning_size - 1, self.board_size[0]):
            rows.append(self.get_diagonal_row(x, 0, -1, x + 1, -1, 1))
        for y in range(1, self.board_size[1] - self.winning_size + 1):
            rows.append(self.get_diagonal_row(self.board_size[0] - 1, y, y - 1, self.board_size[1], -1, 1))
        for x in range(self.board_size[0] - self.winning_size, 0, -1):
            rows.append(self.get_diagonal_row(x, 0, self.board_size[0], self.board_size[1] - x, 1, 1))
        for y in range(self.board_size[1] - self.winning_size + 1):
            rows.append(self.get_diagonal_row(0, y, self.board_size[0] - y, self.board_size[1], 1, 1))
        return rows

    def eval_state(self):
        for x in range(self.board_size[0]):
            val1 = self.eval_row(self.state[x, :])
            if val1 in self.terminal_point_states:
                return val1
        for y in range(self.board_size[1]):
            val2 = self.eval_row(self.state[:, y])
            if val2 in self.terminal_point_states:
                return val2
        for row in self.get_diagonal_rows():
            val3 = self.eval_row(row)
            if val3 in self.terminal_point_states:
                return val3
        return 0

    def is_ended(self):
        return self.eval_state() != 0 or len(self.get_available_points()[0]) == 0

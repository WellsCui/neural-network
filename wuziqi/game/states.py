import numpy as np
import random

BOARD_SIZE = (11, 11)
WINNING_SIZE = 5
START_START = np.zeros(BOARD_SIZE)
POINT_STATES = [-1, 0, 1]
TERMINAL_POINT_STATES = [-1, 1]


def random_state():
    s = [random.randint(POINT_STATES[0], POINT_STATES[-1]) for r in range(121)]
    return np.reshape(s, BOARD_SIZE)


def eval_point_states(points):
    last_val = 0
    repeat_count = 0
    for p in points:
        if p == last_val:
            repeat_count += 1
            if (p in TERMINAL_POINT_STATES) and repeat_count >= WINNING_SIZE:
                return p
        else:
            last_val = p
            repeat_count = 0
    return 0


def get_triangle_row(state, start_x, start_y, end_x, end_y, x_step, y_step):
    # print("start_x, end_x, x_step, start_y, end_y, y_step:", start_x, end_x, x_step, start_y, end_y, y_step)
    row = state[range(start_x, end_x, x_step), range(start_y, end_y, y_step)]

    # print(row)
    return row


def get_triangle_rows(state):
    rows = []
    for x in range(WINNING_SIZE-1, BOARD_SIZE[0]):
        rows.append(get_triangle_row(state, x, 0, -1, x+1, -1, 1))
    for y in range(1, BOARD_SIZE[1]-WINNING_SIZE+1):
        rows.append(get_triangle_row(state, BOARD_SIZE[0]-1, y, y-1, BOARD_SIZE[1], -1, 1))
    for x in range(BOARD_SIZE[0]-WINNING_SIZE, 0, -1):
        rows.append(get_triangle_row(state, x, 0, BOARD_SIZE[0], BOARD_SIZE[1]-x, 1, 1))
    for y in range(BOARD_SIZE[1]-WINNING_SIZE+1):
        rows.append(get_triangle_row(state, 0, y, BOARD_SIZE[0]-y, BOARD_SIZE[1], 1, 1))
    return rows


def eval_state(state):
    for x in range(BOARD_SIZE[0]):
        val1 = eval_point_states(state[x, :])
        if val1 in TERMINAL_POINT_STATES:
            print("row match in", x)
            return val1
    for y in range(BOARD_SIZE[1]):
        val2 = eval_point_states(state[:, y])
        if val2 in TERMINAL_POINT_STATES:
            print("col match in", y)
            return val2
    for triangle_row in get_triangle_rows(state):
        val3 = eval_point_states(triangle_row)
        if val3 in TERMINAL_POINT_STATES:
            print("triangle match:", triangle_row)
            return val3
    return 0

# class Agent(object):


# for i in range(10):
#     s = random_state()
#     val = eval_state(s)
#     if val in TERMINAL_POINT_STATES:
#         print("value:", val)
#         print(s)

s = random_state()
print(s)
print(s.reshape((11, 11, 1)))

# print(np.where(s == 0))
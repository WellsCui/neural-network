import game.wuziqi as wuziqi
import numpy as np
import re


def build_session(moves):
    board_size = (15, 15)
    side = 1
    player1_state_actions = []
    player2_state_actions = []
    game = wuziqi.WuziqiGame(board_size)

    for step in range(int(len(moves) / 2)):
        # print("Step:", step)
        action = wuziqi.WuziqiAction(int(moves[step * 2], 16), int(moves[step * 2 + 1], 16), side)
        state = game.get_state().copy()
        game.update(action)
        # game.show()
        reward = game.eval_state() * side
        if side == 1:
            player1_state_actions.append([state, action, reward])
        else:
            player2_state_actions.append([state * -1, action.reverse(), reward])
        if reward != 0:
            break
        side *= -1

    final_state = game.eval_state()
    if final_state != 0:
        # player1_state_actions[-1][2] = 1
        player1_state_actions.append([game.get_state().copy(), wuziqi.WuziqiAction(0, 0, 0), 0])
        # elif final_state == -1:
        # player2_state_actions[-1][2] = 1
        player2_state_actions.append([game.get_state() * -1, wuziqi.WuziqiAction(0, 0, 0), 0])
    return player1_state_actions, player2_state_actions, game.eval_state()


def get_sessions(bdt_file, max_session=None):
    with open(bdt_file) as f:
        lines = f.readlines()
    # sessions = []
    i = 0
    for game in lines:
        # print("game script:", game)
        if (max_session is not None) and i > max_session:
            break
        try:
            matches = re.search('\[(?P<player1>\S+),(?P<player2>\S+),(?P<winner>[+-=]),(?P<moves>\w+),', game)
            # print("Player1: %s, Player2: %s, Winner: %s" % (matches.group('player1'),
            #                                                 matches.group('player2'),
            #                                                 matches.group('winner')))
            moves = matches.group('moves')

            if moves.startswith('1001300320024005600650041'):
                continue


            # if i % 10 == 0:
            #     input('Press enter to continue:')

            yield build_session(moves)
            i += 1
        except:
            continue


def replay_game(moves):
    # moves = '88FFFE98798A6A975B4C59999A7BA86C5D5C3C7A4B896BA7B6A99687'
    board_size = (15, 15)
    game = wuziqi.WuziqiGame(board_size)
    side = 1
    for step in range(int(len(moves) / 2)):
        game.update(wuziqi.WuziqiAction(int(moves[step * 2], 16), int(moves[step * 2 + 1], 16), side))
        game.show()
        side *= -1
    return game.eval_state()


def replay_games(bdt_file):
    # RHL2006.bdt
    with open(bdt_file) as f:
        lines = f.readlines()
    result = np.zeros(len(lines))
    i = 0
    for game in lines:
        print(i, " game script:", game)
        # 1997,382=[marik,om-by,-,88FFFE98798A6A975B4C59999A7BA86C5D5C3C7A4B896BA7B6A99687,?,?]
        matches = re.search('\[(?P<player1>\S+),(?P<player2>\S+),(?P<winner>[+-=]),(?P<moves>\w+),', game)
        print("Player1: %s, Player2: %s, Winner: %s" % (matches.group('player1'),
                                                        matches.group('player2'),
                                                        matches.group('winner')))
        moves = matches.group('moves')

        result[i] = replay_game(moves)
        if i % 2 == 0:
            input('Press any key to continue:')
        i += 1
    print("Win: %d, Loss: %d, Unfinished: %d" %
          (len(np.where(result == 1)[0]), len(np.where(result == -1)[0]), len(np.where(result == 0)[0])))

def replay_sessions():
    for s in get_sessions('data/gomocup-2016.bdt'):
        print("****************************************")

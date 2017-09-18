import game.wuziqi as wuziqi
import os


def replay_psq_game(game_file):
    with open(game_file) as f:
        lines = f.readlines()
    board_size = (15, 15)
    game = wuziqi.WuziqiGame(board_size)
    side = 1
    for move in lines[1:-3]:
        x, y, _ = move.split(',')
        game.update(wuziqi.WuziqiAction(int(x) - 1, int(y) - 1, side))
        game.show()
        side *= -1
    return game.eval_state()


def rebuild_psq_game(game_file):
    with open(game_file) as f:
        lines = f.readlines()
    board_size = (15, 15)
    player1_state_actions = []
    player2_state_actions = []
    game = wuziqi.WuziqiGame(board_size)
    side = 1
    for move in lines[1:-3]:
        x, y, _ = move.split(',')
        action = wuziqi.WuziqiAction(int(x) - 1, int(y) - 1, side)
        data = [game.get_state().copy(), action, 0]
        game.update(action)
        if side == 1:
            player1_state_actions.append(data)
        else:
            player2_state_actions.append(data)
        side *= -1

    final_state = game.eval_state()
    if final_state == 1:
        player1_state_actions[-1][2] = 1
        player1_state_actions.append([game.get_state().copy(), wuziqi.WuziqiAction(0, 0, 0), 0])
    elif final_state == -1:
        player2_state_actions[-1][2] = 1
        player2_state_actions.append([game.get_state().copy(), wuziqi.WuziqiAction(0, 0, 0), 0])

    return player1_state_actions, player2_state_actions, final_state


def replay_psq_games():
    for file in os.listdir("../history/standard"):
        if file.endswith(".psq"):
            replay_psq_game(os.path.join("../history/standard", file))

def convert_psq_to_bdt(psq_file):
    with open(psq_file) as f:
        lines = f.readlines()
    player1 = "player1"
    player2 = "player2"
    date = '2017,1'
    winning = '+'
    moves = []
    hex_str='0123456789ABCDEF'
    # 1997,382=[marik,om-by,-,88FFFE98798A6A975B4C59999A7BA86C5D5C3C7A4B896BA7B6A99687,?,?]

    for move in lines[1:-3]:
        x, y, _ = move.split(',')
        moves.append(hex_str[int(x) - 1]+hex_str[int(y) - 1])
    return '%s=[%s,%s,%s,%s,?,?]\n' % (date, player1, player2, winning, ''.join(moves))


def convert_psq_files(psq_dir, bdt_file):
    games = []
    for file in os.listdir(psq_dir):
        if file.endswith(".psq"):
            try:
                games.append(convert_psq_to_bdt(os.path.join(psq_dir, file)))
            except:
                continue

    with open(bdt_file, 'a') as f:
        for game in games:
            f.write(game)


def get_training_sessions():
    sessions = []
    for file in os.listdir("data/standard"):
        if file.endswith(".psq"):
            sessions.append(rebuild_psq_game(os.path.join("data/standard", file)))
    return sessions




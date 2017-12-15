import game.alpha_zero.alpha_zero_net as azn
import game.alpha_zero.queued_evaluator as queued_evaluator
import game.alpha_zero.monte_carlo_tree_search as mcts
import game.wuziqi as wuziqi
import numpy as np
import datetime


def self_play(mts: mcts.McTreeSearch):
    state, p = mts.move()
    steps = []
    steps.append((state, p))
    c = 0
    while not wuziqi.WuziqiGame(mts.net.board_size, state).is_ended():
        c += 1
        if c > 10 and mts.options.temperature == 1:
            mts.options.temperature = 1e-8
        state, p = mts.move()
        steps.append((state, p))
    return steps


def run(train_times, game_count, model_path):
    board_size = (15, 15)
    init_state = np.zeros(board_size)
    net = azn.AlphaZeroNet('AlphaZeroNet', board_size, 0.0005, 0.99)
    net.restore(model_path)
    evaluator = queued_evaluator.QueuedEvaluator(net, 10)
    evaluator.start()
    mts = mcts.McTreeSearch(mcts.McNode(init_state, -1), net, evaluator, mcts.McTreeSearchOption(100, 1, 0.5))
    games = []
    for t in range(train_times):
        for i in range(game_count):
            begin = datetime.datetime.now()
            games.append(mts.self_play())
            end = datetime.datetime.now()
            print('game %d completed in %d seconds' % (i, (end-begin).seconds))
            mts.set_root(mcts.McNode(init_state, -1))
        states = []
        probabilities = []
        vals = []
        for g in games:
            for s, p, v in g:
                states.append(s)
                probabilities.append(p)
                vals.append(v)

        net.train((states, probabilities, vals), 100)
        net.save(model_path)


# run(2, 2)
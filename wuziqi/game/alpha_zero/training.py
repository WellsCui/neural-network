import game.alpha_zero.alpha_zero_net as azn
import game.alpha_zero.queued_evaluator as queued_evaluator
import game.alpha_zero.monte_carlo_tree_search as mcts
import game.wuziqi as wuziqi
import numpy as np


def self_play(mts: mcts.McTreeSearch):
    state, p = mts.play()
    steps = []
    steps.append((state, p))
    c = 0
    while not wuziqi.WuziqiGame(mts.net.board_size, state).is_ended():
        c += 1
        if c > 10 and mts.options.temperature == 1:
            mts.options.temperature = 1e-8
        state, p = mts.play()
        steps.append((state, p))
    return steps

def run():
    board_size = (15, 15)
    init_state = np.zeros(board_size)
    net = azn.AlphaZeroNet('AlphaZeroNet', board_size, 0.0005, 0.99)
    evaluator = queued_evaluator.QueuedEvaluator(net, 10)
    evaluator.start()
    mts = mcts.McTreeSearch(mcts.McNode(init_state, -1), net, evaluator, mcts.McTreeSearchOption(10, 1, 0.5))
    self_play(mts)


run()
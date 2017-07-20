import game.wuziqi as wuziqi
import game.agents as agents
import game.evaluators as evaluators
import numpy as np
import time


def play(times):
    player1 = agents.WuziqiRandomAgent(1)
    player2 = agents.WuziqiRandomAgent(-1)

    for i in range(times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame((11, 11))
        step = 0
        sleep_time = 0
        while True:
            step += 1
            action = player1.act(game)
            print("player1 toke action:", action.x, action.y)
            game.show()
            time.sleep(sleep_time)
            if game.is_ended():
                print("game ended after player1 toke action:", action.x, action.y)
                break

            action = player2.act(game)
            print("player2 toke action:", action.x, action.y)
            game.show()
            time.sleep(sleep_time)
            if game.is_ended():
                print("game ended after player2 toke action:", action.x, action.y)
                break
        print("Game is ended on step:", step)
        val = wuziqi.WuziqiGame.eval_state(game.board_size, game.state)
        if val == 1:
            winner = "player1"
        elif val == -1:
            winner = "player2"
        else:
            winner = "nobody"
        print(" winner is", winner)
        # game.show()


# play(10)


def train_evaluator(game_times, validate_frequency):
    player1 = agents.WuziqiRandomAgent(1)
    player2 = agents.WuziqiRandomAgent(-1)
    evaluator = evaluators.WuziqiEvaluator(0.95, (11, 11), 100, 0.001, 1)
    for i in range(game_times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame((11, 11))
        states = [np.copy(game.get_state())]
        step = 0
        while True:
            step += 1
            action = player1.act(game)
            states.append(np.copy(game.get_state()))
            # print(game.get_state())
            # print("player1 take action:", action.x, action.y)
            if game.is_ended():
                break
            action = player2.act(game)
            states.append(np.copy(game.get_state()))
            # print(game.get_state())
            # print("player2 take action:", action.x, action.y)
            if game.is_ended():
                break
        # print(states)
        # print("Game is ended on step:", step)

        # print("state count:", len(states))
        val = wuziqi.WuziqiGame.eval_state(game.board_size, game.state)
        if val == 1:
            winner = "player1"
        elif val == -1:
            winner = "player2"
        else:
            winner = "nobody"

        if i % validate_frequency > 0:
            evaluator.train(states)
            print(" winner is", winner, "predict after training :", evaluator.predict(states[-2:]))
        else:
            preds = evaluator.predict(states[-2:])
            print(" winner is", winner, "predict without training:", preds)
        # game.show()



train_evaluator(1001, 10)

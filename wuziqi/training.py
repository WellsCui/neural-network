import game.wuziqi as wuziqi
import game.agents as agents
import game.evaluators as evaluators
import numpy as np
import time
import game.actor_critic_agent as ac_agent
import game.competing_agent as competing_agent


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


def train_evaluator(game_times, validate_times):
    player1 = agents.WuziqiRandomAgent(1)
    player2 = agents.WuziqiRandomAgent(-1)
    game_size = (15, 15)
    evaluator = evaluators.WuziqiEvaluator(0.95, game_size, 100, 0.001, 1)
    games = []
    for i in range(game_times+validate_times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame(game_size)
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
        if i < game_times:
            games.append(states)
            if i % 10 == 0:
                print("training with 10 games ...")
                evaluator.train(games)
                games.clear()
            if i % 20 == 0 or i % 23 == 0:
                val = wuziqi.WuziqiGame.eval_state(game.board_size, game.state)
                if val == 1:
                    winner = "player1"
                elif val == -1:
                    winner = "player2"
                else:
                    winner = "nobody"
                preds = evaluator.predict(states[-2:])
                print(" winner is", winner, "predict without training:", preds)

        else:
            val = wuziqi.WuziqiGame.eval_state(game.board_size, game.state)
            if val == 1:
                winner = "player1"
            elif val == -1:
                winner = "player2"
            else:
                winner = "nobody"
            preds = evaluator.predict(states[-2:])
            print(" winner is", winner, "predict without training:", preds)


        # print(states)
        # print("Game is ended on step:", step)

        # print("state count:", len(states))


        # if i % validate_frequency > 0:
        #     evaluator.train(states)
        #     # print(" winner is", winner, "predict after training :", evaluator.predict(states[-2:]))
        # else:
        #     preds = evaluator.predict(states[-2:])
        #     print(" winner is", winner, "predict without training:", preds)
        # game.show()


def get_reward(game):
    return wuziqi.WuziqiGame.eval_state(game.board_size, game.get_state())


def run_actor_critic_agent(times):
    player1 = ac_agent.ActorCriticAgent((11, 11), 0.001, 1, 0.95)
    player2 = agents.WuziqiRandomAgent(-1)

    # q_net = ac_agent.WuziqiQValueNet((11, 11), 0.001, 1, 0.95)
    # q_net.build_state_action(game.get_state(), wuziqi.WuziqiAction(1, 1, 1))
    wins = 0
    for i in range(times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame((11, 11))
        step = 0
        current_state = game.get_state()
        current_action = player1.act(game)

        while True:
            step += 1
            r = get_reward(game)
            if game.is_ended():
                next_state = game.get_state()
                next_action = wuziqi.WuziqiAction(0, 0, 0)
                player1.learn(current_state, current_action, r, next_state, next_action)
                break
            else:
                player2.act(game)
                next_state = game.get_state()
                if game.is_ended():
                    game.show()
                    next_action = wuziqi.WuziqiAction(0, 0, 0)
                    player1.learn(current_state, current_action, get_reward(game), next_state, next_action)
                    break
                else:
                    next_action = player1.act(game)
            game.show()
            player1.learn(current_state, current_action, r, next_state, next_action)
            current_action = next_action
            current_state = next_state

        print("Game is ended on step:", step)
        val = wuziqi.WuziqiGame.eval_state(game.board_size, game.state)
        if val == 1:
            wins += 1
            winner = "player1 " + str(wins) + " out of " + str(i+1)

        elif val == -1:
            winner = "player2 " + str(i+1 - wins) + " out of " + str(i+1)
        else:
            winner = "nobody"
        print(" winner is", winner)
        # game.show()


def run_competing_agent(times, train_all_after, restored):
    board_size = (11, 11)
    player1 = competing_agent.CompetingAgent("player1", board_size, 0.005, 1, 0.985)
    player2 = competing_agent.CompetingAgent("player2", board_size, 0.005, -1, 0.985)

    player1.is_greedy = True
    player2.is_greedy = True
    # if restored:
    #     player1.restore("/tmp/player1")
    #     player1.restore("/tmp/player2")

    wins = 0

    # train_all_after = 1

    def move(player, game):
        action = player.act(game)
        game.show()
        return game.is_ended()

    for i in range(times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame(board_size)
        step = 0
        if i > train_all_after:
            player1.train_sessions_with_end_only = False
            player2.train_sessions_with_end_only = False
        if i % 2 == 0:
            first_player = player1
            second_player = player2
        else:
            first_player = player2
            second_player = player1
        # player1.is_greedy = i % 11 == 0
        # player2.is_greedy = i % 13 == 0

        print("player1 greedy: ", player1.is_greedy)
        print("player2 greedy: ", player2.is_greedy)
        while True:
            step += 1
            if move(first_player, game):
                break
            if move(second_player, game):
                break

        print("Game is ended on step:", step)
        val = wuziqi.WuziqiGame.eval_state(game.board_size, game.state)
        if val == 1:
            wins += 1
            winner = "player1 " + str(wins) + " out of " + str(i+1)
            player1.increase_greedy()
        elif val == -1:
            winner = "player2 " + str(i+1 - wins) + " out of " + str(i+1)
            player2.increase_greedy()
        else:
            winner = "nobody"
        print("Winner is", winner)
        # player1.save("/tmp/player1")
        # player1.save("/tmp/player2")


run_competing_agent(1000, 200, False)


# train_evaluator(1000, 100)
#run_actor_critic_agent(10)

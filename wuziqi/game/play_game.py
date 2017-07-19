import game.wuziqi as wuziqi
import game.agents as agents


def play(times):
    player1 = agents.WuziqiRandomAgent(1)
    player2 = agents.WuziqiRandomAgent(-1)

    for i in range(times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame((11, 11), 5)
        step = 0
        while True:
            step += 1
            action = player1.act(game)
            # print("player1 take action:", action.x, action.y)
            if game.is_ended():
                break
            action = player2.act(game)
            # print("player2 take action:", action.x, action.y)
            if game.is_ended():
                break
        print("Game is ended on step:", step)
        val = game.eval_state()
        if val == 1:
            winner = "player1"
        elif val == -1:
            winner = "player2"
        else:
            winner = "nobody"
        print(" winner is", winner)
        print(game.get_state())


import game.interfaces as interfaces
import game.wuziqi as wuziqi
import sys


class HumanAgent(interfaces.IAgent):
    def __init__(self, name, board_size, side):
        self.name = name
        self.side = side
        self.board_size = board_size

    def act(self, environment: interfaces.IEnvironment):
        action = None
        while action is None:
            try:
                pos = eval(input('Enter your next move:'))
                if pos != -1:
                    if environment.get_state()[pos[1], pos[0]] != 0:
                        print("Your move is invalid.")
                    else:
                        action = wuziqi.WuziqiAction(pos[0], pos[1], self.side)
            except:
                print("Your input is invalid.")
                continue
            if pos == -1:
                sys.exit("Game is terminated.")
        environment.update(action)
        return action

    def learn_from_experience(self, experience, learn_from_winner):
        pass

    def save_model(self, model_dir):
        pass

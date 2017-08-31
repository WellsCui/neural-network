import game.interfaces as interfaces
import game.wuziqi as wuziqi


class HumanAgent(interfaces.IAgent):
    def __init__(self, name, side):
        self.name = name
        self.side = side

    def act(self, environment: interfaces.IEnvironment):
        action = None
        while action is None:
            try:
                pos = eval(input('Enter your next move:'))
                if environment.get_state()[pos[0], pos[1]] != 0:
                    print("Your move is invalid.")
                else:
                    action = wuziqi.WuziqiAction(pos[0], pos[1], self.side)
            except:
                print("Your input is invalid.")
                continue

        environment.update(action)
        return action

    def learn_from_experience(self, experience, learn_from_winner):
        pass

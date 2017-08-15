import random
import game.interfaces as interfaces
import game.wuziqi as wuziqi


class WuziqiRandomAgent(interfaces.IAgent):
    def __init__(self, side):
        self.side = side

    def act(self, environment: interfaces.IEnvironment):
        points = environment.get_available_points()
        i = random.randint(0, len(points[0]) - 1)
        action = wuziqi.WuziqiAction(points[0][i], points[1][i], self.side)
        environment.update(action)
        return action


class WuziqiPolicyAgent(interfaces.IAgent):
    def __init__(self, policy: interfaces.IPolicy):
        self.policy = policy

    def act(self, environment: interfaces.IEnvironment):
        action = self.policy.suggest(environment.get_state())
        environment.update(action)
        return action



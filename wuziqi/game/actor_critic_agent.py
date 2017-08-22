import game.wuziqi_value_net as qlearning
import game.interfaces as interfaces

class ActorCriticAgent(interfaces.IAgent):
    def __init__(self, board_size, learning_rate, side, lbd):
        self.side = side
        self.policy = qlearning.WuziqiPolicyNet(board_size, learning_rate, lbd)
        self.qnet = qlearning.WuziqiQValueNet(board_size, learning_rate, lbd)
        self.mode = "online_learning."
        self.lbd = lbd

    def act(self, environment: interfaces.IEnvironment):
        action = self.policy.suggest(environment.get_state(), self.side, 1)[0]
        environment.update(action)
        return action

    def learn(self, current_state, current_action, r, next_state, next_action):
        qnet_value = self.qnet.evaluate(current_state, current_action)[0, 0]
        corrected_qnet_value = r + self.lbd * self.qnet.evaluate(next_state, next_action)[0, 0]
        print("Estimate : ", qnet_value,
              " reward: ", r,
              " Corrected", corrected_qnet_value)
        self.qnet.apply_gradient(current_state, current_action, r, next_state, next_action)
        self.policy.apply_gradient(current_state, current_action, corrected_qnet_value)

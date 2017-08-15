import game.qlearning as qlearning
import numpy as np
import game.interfaces as interfaces
import game.wuziqi as wuziqi


class CompetingAgent(interfaces.IAgent):
    def __init__(self, board_size, learning_rate, side, lbd):
        self.side = side
        self.policy = qlearning.WuziqiPolicyNet(board_size, learning_rate, lbd)
        self.qnet = qlearning.WuziqiQValueNet(board_size, learning_rate, lbd)
        self.mode = "online_learning."
        self.lbd = lbd
        self.think_depth = 10
        self.think_width = 10
        self.policy_trainning_data = []
        self.policy_trainning_size = 50

    def act(self, environment: interfaces.IEnvironment):
        state = environment.get_state().copy()
        actions = self.policy.suggest(state, self.side, self.think_width)
        rehearsals = []

        for act in actions:
            rehearsals.append(self.rehearsal(environment.clone(), act, self.policy, self.think_depth))

        index = np.argmax([self.evalue_rehearsal(rehearsal) for rehearsal in rehearsals])
        self.learn_from_rehearsal(state, rehearsals, actions[index])
        environment.update(actions[index])
        return actions[index]

    def learn(self, current_state, current_action, r, next_state, next_action):
        qnet_value = self.qnet.evaluate(current_state, current_action)[0, 0]
        corrected_qnet_value = r + self.lbd * self.qnet.evaluate(next_state, next_action)[0, 0]
        print("Estimate : ", qnet_value,
              " reward: ", r,
              " Corrected", corrected_qnet_value)
        self.qnet.apply_gradient(current_state, current_action, r, next_state, next_action)
        self.policy.apply_gradient(current_state, current_action, corrected_qnet_value)

    def learn_from_rehearsal(self, state, rehearsals, chosen_action):
        history = [h1 for h1, h2 in rehearsals]
        self.qnet.train(history)
        self.policy_trainning_data.append([state, chosen_action])
        if len(self.policy_trainning_data) == self.policy_trainning_size:
            self.policy.train(self.policy_trainning_data)
            self.policy_trainning_data = []

    def evalue_rehearsal(self, rehearsal):
        history, opponent_history = rehearsal
        state, action, reward = history[-1]
        return self.qnet.evaluate(state, action)

    def rehearsal(self, environment: interfaces.IEnvironment, action, opponent_policy: interfaces.IPolicy, steps):
        def get_reward(side):
            return wuziqi.WuziqiGame.eval_state(environment.board_size, environment.get_state()) * side

        history1 = []
        history2 = []
        next_state_1 = environment.get_state().copy()
        next_action_1 = action

        while steps > 0:
            steps -= 1
            environment.update(next_action_1)
            r = get_reward(self.side)
            history1.append([next_state_1, next_action_1, r])
            if environment.is_ended():
                next_state_1 = environment.get_state().copy()
                next_action_1 = wuziqi.WuziqiAction(0, 0, 0)
                history1.append([next_state_1, next_action_1, 0])
                history2.append([next_state_1*-1, next_action_1, 0])
                break
            else:
                next_state_2 = environment.get_state().copy() * -1
                next_action_2 = opponent_policy.suggest(next_state_2, self.side * -1, 1)[0]
                next_state_1 = environment.update(next_action_2)
                r = get_reward(self.side * -1)
                history2.append([next_state_2, next_action_2, r])

                if environment.is_ended():
                    next_action_1 = wuziqi.WuziqiAction(0, 0, 0)
                    history1.append([next_state_1, next_action_1, 0])
                    history2.append([next_state_1 * -1, next_action_1, 0])
                    break
                else:
                    next_action_1 = self.policy.suggest(environment.get_state(), self.side, 1)[0]
        return history1, history2


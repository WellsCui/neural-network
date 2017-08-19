import game.qlearning as qlearning
import numpy as np
import game.interfaces as interfaces
import game.wuziqi as wuziqi
import game.utils
import random


class CompetingAgent(interfaces.IAgent):
    def __init__(self, board_size, learning_rate, side, lbd):
        self.side = side
        self.policy = qlearning.WuziqiPolicyNet(board_size, learning_rate, lbd)
        self.qnet = qlearning.WuziqiQValueNet(board_size, learning_rate, lbd)
        self.mode = "online_learning."
        self.lbd = lbd
        self.think_depth = 5
        self.think_width = 10
        self.policy_training_data = []
        self.policy_trainning_size = 50
        self.epslon = 0.001
        self.greedy_rate = 0.5
        self.board_size = board_size
        self.is_greedy = False

    def act(self, environment: interfaces.IEnvironment):
        state = environment.get_state().copy()
        # select_count = int(self.think_width * self.greedy_rate)
        select_count = 1
        actions = self.policy.suggest(state, self.side, select_count)
        available_actions = self.get_available_actions(environment, self.side)
        actions += random.sample(available_actions, self.think_width - select_count)
        rehearsals = []

        for act in actions:
            rehearsals.append(self.rehearsal(environment.clone(), act, self.policy, self.think_depth))

        best_choice = np.argmax([self.evaluate_rehearsal(rehearsal) for rehearsal in rehearsals])
        if self.is_greedy:
            choice = best_choice
        else:
            choice = game.utils.partial_random(best_choice, range(self.think_width), self.greedy_rate)

        print("best (%d, %d) value: %f in selection %s, policy (%d, %d) value: %f  actual (%d, %d) value: %f, in selection %s is best %s" %
              (actions[best_choice].x, actions[best_choice].y,
               self.qnet.evaluate(state, actions[best_choice]),
               best_choice < select_count,
               actions[0].x, actions[0].y,
               self.qnet.evaluate(state, actions[0]),
               actions[choice].x, actions[choice].y,
               self.qnet.evaluate(state, actions[choice]),
               choice < select_count,
               choice == best_choice))

        self.learn_from_rehearsal(state, rehearsals, actions[best_choice])

        print("best (%d, %d) value: %f in selection %s, policy (%d, %d) value: %f  actual (%d, %d) value: %f, in selection %s is best %s" %
              (actions[best_choice].x, actions[best_choice].y,
               self.qnet.evaluate(state, actions[best_choice]),
               best_choice < select_count,
               actions[0].x, actions[0].y,
               self.qnet.evaluate(state, actions[0]),
               actions[choice].x, actions[choice].y,
               self.qnet.evaluate(state, actions[choice]),
               choice < select_count,
               choice == best_choice))

        environment.update(actions[choice])
        return actions[choice]

    def increase_greedy(self):
        self.greedy_rate += (1 - self.greedy_rate) * self.epslon
        print("greedy : ", self.greedy_rate)

    def learn(self, current_state, current_action, r, next_state, next_action):
        qnet_value = self.qnet.evaluate(current_state, current_action)[0, 0]
        corrected_qnet_value = r + self.lbd * self.qnet.evaluate(next_state, next_action)[0, 0]
        print("Estimate : ", qnet_value,
              " reward: ", r,
              " Corrected", corrected_qnet_value)
        self.qnet.apply_gradient(current_state, current_action, r, next_state, next_action)
        self.policy.apply_gradient(current_state, current_action, corrected_qnet_value)

    def learn_from_rehearsal(self, state, rehearsals, chosen_action):
        h = []
        for h1, h2, final_state in rehearsals:
            h.append(h1)
            h.append(h2)
        self.qnet.train(h)
        self.policy_training_data.append([state, chosen_action])
        if len(self.policy_training_data) == self.policy_trainning_size:
            self.policy.train(self.policy_training_data)
            self.policy_training_data = []

    def evaluate_rehearsal(self, rehearsal):
        history, opponent_history, final_state = rehearsal
        state, action, reward = history[-1]

        if final_state == 0:
            if len(opponent_history) > 0:
                opponent_state, opponent_action, opponent_reward = opponent_history[-1]
                return self.qnet.evaluate(state, action) - self.qnet.evaluate(opponent_state, opponent_action)
            return self.qnet.evaluate(state, action)
        else:
            return final_state

    @staticmethod
    def get_available_actions(environment: interfaces.IEnvironment, side):
        points = environment.get_available_points()
        actions = []
        for i in range(len(points[0])):
            actions.append(wuziqi.WuziqiAction(points[0][i], points[1][i], side))
        return actions

    def rehearsal(self, environment: interfaces.IEnvironment, action, opponent_policy: interfaces.IPolicy, steps):
        def get_reward(side):
            return wuziqi.WuziqiGame.eval_state(environment.board_size, environment.get_state()) * side

        def get_partial_random_action(best_action, side):
            return game.utils.partial_random(best_action, self.get_available_actions(environment, side), self.greedy_rate)

        history1 = []
        history2 = []
        next_state_1 = environment.get_state().copy()
        next_action_1 = action
        final_state = 0

        while steps > 0:
            steps -= 1
            environment.update(next_action_1)
            r1 = get_reward(self.side)
            history1.append([next_state_1, next_action_1, r1])
            if environment.is_ended():
                next_state_1 = environment.get_state().copy()
                next_action_1 = wuziqi.WuziqiAction(0, 0, 0)
                history1.append([next_state_1, next_action_1, 0])
                # history1.append([np.zeros(environment.board_size), wuziqi.WuziqiAction(0, 0, 0), 0])
                if len(history2) > 0:
                    # history2[-1][-1] = -1
                    history2.append([next_state_1 * -1, wuziqi.WuziqiAction(0, 0, 0), 0])
                final_state = 1
                break
            else:
                next_state_2 = environment.reverse().get_state().copy()
                next_action_2 = get_partial_random_action(opponent_policy.suggest(next_state_2, self.side * -1, 1)[0],
                                                          self.side * -1)

                state = environment.update(next_action_2).copy()
                r2 = get_reward(self.side * -1)
                history2.append([next_state_2, next_action_2.reverse(), r2])

                if environment.is_ended():
                    # history1[-1][-1] = -1
                    next_action_1 = wuziqi.WuziqiAction(0, 0, 0)
                    history1.append([state, next_action_1, 0])
                    history2.append([state * -1, next_action_2.reverse(), 0])
                    # history2.append([np.zeros(environment.board_size), wuziqi.WuziqiAction(0, 0, 0), 0])
                    final_state = -1
                    break
                else:
                    next_state_1 = state
                    next_action_1 = get_partial_random_action(self.policy.suggest(state, self.side, 1)[0], self.side)
        return history1, history2, final_state

    def save(self, save_dir):
        self.qnet.save(save_dir)
        self.policy.save(save_dir)

    def restore(self, restore_dir):
        self.qnet.restore(restore_dir)
        self.policy.restore(restore_dir)

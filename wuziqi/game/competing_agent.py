import game.qlearning as qlearning
import numpy as np
import game.interfaces as interfaces
import game.wuziqi as wuziqi
import game.utils
import random


class CompetingAgent(interfaces.IAgent):
    def __init__(self, name, board_size, learning_rate, side, lbd):
        self.name = name
        self.side = side
        self.policy = qlearning.WuziqiPolicyNet(board_size, learning_rate, lbd)
        self.qnet = qlearning.WuziqiQValueNet(board_size, learning_rate, lbd)
        self.mode = "online_learning."
        self.lbd = lbd
        self.search_depth = 20
        self.search_width = 20
        self.policy_training_data = []
        self.value_net_training_data = []
        self.value_net_trainning_size = 50
        self.policy_trainning_size = 50
        self.epslon = 0.001
        self.greedy_rate = 0.5
        self.board_size = board_size
        self.is_greedy = False
        self.train_sessions_with_end_only = True
        self.last_action = None

    def act(self, environment: interfaces.IEnvironment):
        state = environment.get_state().copy()
        # select_count = int(self.think_width * self.greedy_rate)
        select_count = 1
        actions = self.policy.suggest(state, self.side, select_count)
        available_actions = self.get_available_actions(environment, self.side)
        actions += random.sample(available_actions, self.search_width - select_count)
        rehearsals = []

        for act in actions:
            rehearsals.append(self.rehearsal(environment.clone(), act, self.policy, self.search_depth))

        best_choice = np.argmax([self.evaluate_rehearsal(rehearsal) for rehearsal in rehearsals])
        if self.is_greedy:
            choice = best_choice
        else:
            choice = game.utils.partial_random(best_choice, range(self.search_width), self.greedy_rate)

        print("%s: best (%d, %d): %f in policy %s, policy (%d, %d): %f actual (%d, %d): %f, is best %s" %
              (self.name,
               actions[best_choice].x, actions[best_choice].y,
               self.qnet.evaluate(state, actions[best_choice]),
               best_choice < select_count,
               actions[0].x, actions[0].y,
               self.qnet.evaluate(state, actions[0]),
               actions[choice].x, actions[choice].y,
               self.qnet.evaluate(state, actions[choice]),
               choice == best_choice))

        if self.learn_from_rehearsal(rehearsals, best_choice):
            print("best (%d, %d): %f in policy %s, policy (%d, %d): %f actual (%d, %d): %f, is best %s" %
                  (actions[best_choice].x, actions[best_choice].y,
                   self.qnet.evaluate(state, actions[best_choice]),
                   best_choice < select_count,
                   actions[0].x, actions[0].y,
                   self.qnet.evaluate(state, actions[0]),
                   actions[choice].x, actions[choice].y,
                   self.qnet.evaluate(state, actions[choice]),
                   choice == best_choice))

        self.last_action = actions[choice]
        environment.update(self.last_action)
        return self.last_action

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

    def learn_from_rehearsal(self, rehearsals, best_choice):

        for h1, h2, final_state in rehearsals:
            if self.train_sessions_with_end_only:
                if final_state == 1:
                    self.value_net_training_data.append(h1)
                elif final_state == -1:
                    self.value_net_training_data.append(h2)
            else:
                self.value_net_training_data.append(h1)
                self.value_net_training_data.append(h2)

        if len(self.value_net_training_data) >= self.value_net_trainning_size:
            self.qnet.train(self.value_net_training_data)
            self.value_net_training_data = []
            return True

        best_rehearsal = rehearsals[best_choice]
        state, best_action, _ = best_rehearsal[0][0]
        if self.train_sessions_with_end_only:
            last_state, last_action, last_reward = best_rehearsal[0][-1]
            if last_action.val == 0:
                # print("add policy training record:", len(self.policy_training_data))
                self.policy_training_data.append([state, best_action])
        else:
            self.policy_training_data.append([state, best_action])

        if len(self.policy_training_data) == self.policy_trainning_size:
            self.policy.train(self.policy_training_data)
            self.policy_training_data = []
        return False

    def evaluate_rehearsal(self, rehearsal):
        history, opponent_history, final_state = rehearsal
        state, action, reward = history[-1]

        if final_state == 0:
            result = self.qnet.evaluate(state, action)
            if len(opponent_history) > 0:
                opponent_state, opponent_action, opponent_reward = opponent_history[-1]
                result = result - self.qnet.evaluate(opponent_state, opponent_action)
        else:
            result = final_state
        return result * (self.lbd ** len(history))

    @staticmethod
    def get_available_actions(environment: interfaces.IEnvironment, side):
        points = environment.get_available_points()
        actions = []
        for i in range(len(points[0])):
            actions.append(wuziqi.WuziqiAction(points[0][i], points[1][i], side))
        return actions

    def get_candidate_actions(environment: interfaces.IEnvironment):
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
                    history2.append([state * -1, next_action_1, 0])
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

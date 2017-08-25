import random
import numpy as np
import game.interfaces as interfaces
import game.wuziqi as wuziqi
import game.utils

import game.wuziqi_value_net
import game.wuziqi_policy_net

class CompetingAgent(interfaces.IAgent):
    def __init__(self, name, board_size, initial_learning_rate, side, lbd):
        self.name = name
        self.side = side
        self.value_net_learning_rate = initial_learning_rate
        self.policy_net_learning_rate = initial_learning_rate
        self.minimum_learning_rate = 0.0002
        self.learning_rate_dacade_rate = 0.6
        self.policy = game.wuziqi_policy_net.WuziqiPolicyNet(board_size, initial_learning_rate, lbd)
        self.qnet = game.wuziqi_value_net.WuziqiQValueNet(board_size, initial_learning_rate, lbd)
        self.mode = "online_learning."
        self.lbd = lbd
        self.search_depth = 20
        self.search_width = 20
        self.policy_training_data = []
        self.value_net_training_data = []
        self.value_net_training_size = 50
        self.policy_training_size = 200
        self.epsilon = 0.001
        self.greedy_rate = 0.5
        self.board_size = board_size
        self.is_greedy = False
        self.last_action = None
        self.qnet_error = 1
        self.policy_accuracy = [0, 0, 0]

    def act(self, environment: interfaces.IEnvironment):
        state = environment.get_state().copy()
        policy_actions, actions = self.get_candidate_actions(environment, self.last_action)
        rehearsals = []
        reversed_rehearsals = []

        for act in actions:
            rehearsals.append(self.rehearsal(environment.clone(), act, self.policy, self.search_depth))
            reversed_rehearsals.append(self.rehearsal(environment.reverse(), act, self.policy, self.search_depth))

        best_action = self.choose_best_action_from_rehearsals(rehearsals, reversed_rehearsals)

        if self.qnet_error < 0.0005:
            self.policy_training_data.append([state, best_action])

        if self.is_greedy:
            action = best_action
        else:
            action = game.utils.partial_random(best_action, actions, self.greedy_rate)

        print("%s: best (%d, %d): %f, in policy: %s, actual (%d, %d): %f best: %s" %
              (self.name,
               best_action.x, best_action.y,
               self.qnet.evaluate(state, best_action),
               self.contain_action(policy_actions, best_action),
               action.x, action.y,
               self.qnet.evaluate(state, action),
               action == best_action))
        self.learn_from_rehearsal(rehearsals+reversed_rehearsals)
        self.last_action = action
        environment.update(self.last_action)
        return self.last_action

    def increase_greedy(self):
        self.greedy_rate += (1 - self.greedy_rate) * self.epsilon
        print("greedy : ", self.greedy_rate)

    def learn(self, current_state, current_action, r, next_state, next_action):
        qnet_value = self.qnet.evaluate(current_state, current_action)[0, 0]
        corrected_qnet_value = r + self.lbd * self.qnet.evaluate(next_state, next_action)[0, 0]
        print("Estimate : ", qnet_value,
              " reward: ", r,
              " Corrected", corrected_qnet_value)
        self.qnet.apply_gradient(current_state, current_action, r, next_state, next_action)
        self.policy.apply_gradient(current_state, current_action, corrected_qnet_value)

    def choose_best_action_from_rehearsals(self, rehearsals, reversed_rehearsals):
        rs = rehearsals+reversed_rehearsals
        vals = [self.evaluate_rehearsal(item) for item in rs]
        index = np.argmax(vals)
        instance = rs[index]
        return instance[0][0][1]

    def learn_from_rehearsal(self, rehearsals):
        self.train_value_net(rehearsals)
        self.train_policy_net(rehearsals)

    def train_policy_net(self, rehearsals):
        for h1, h2, final_state in rehearsals:
            if final_state == 1:
                state, action, _ = h1[-2]
                self.policy_training_data.append([state, action])
            elif final_state == -1:
                state, action, _ = h2[-2]
                self.policy_training_data.append([state, action])

        print("policy_training_data:", len(self.policy_training_data))

        if len(self.policy_training_data) > self.policy_training_size:
            self.policy_accuracy = self.policy.train(self.policy_net_learning_rate, self.policy_training_data)
            if self.policy_net_learning_rate > self.minimum_learning_rate:
                self.policy_net_learning_rate *= self.learning_rate_dacade_rate
            self.policy_training_data = []

    def train_value_net(self, rehearsals):
        for h1, h2, final_state in rehearsals:
            if self.qnet_error > 0.001:
                if final_state == 1:
                    self.value_net_training_data.append(h1)
                elif final_state == -1:
                    self.value_net_training_data.append(h2)
            else:
                self.value_net_training_data.append(h1)
                self.value_net_training_data.append(h2)

        if len(self.value_net_training_data) >= self.value_net_training_size:
            error = self.qnet.train(self.value_net_learning_rate, self.value_net_training_data)
            if error > 0:
                if error < self.learning_rate_dacade_rate * self.qnet_error and self.value_net_learning_rate > self.minimum_learning_rate:
                    self.value_net_learning_rate *= self.learning_rate_dacade_rate
                    self.qnet_error = error

            self.value_net_training_data = []

    def evaluate_rehearsal(self, rehearsal):
        history, opponent_history, final_state = rehearsal
        state, action, reward = history[-1]

        if final_state == 0:
            result = self.qnet.evaluate(state, action)
            if len(opponent_history) > 0:
                opponent_state, opponent_action, opponent_reward = opponent_history[-1]
                result -= self.qnet.evaluate(opponent_state, opponent_action)
        else:
            result = final_state
        return result * (self.lbd ** (len(history) - 2))

    @staticmethod
    def get_available_actions(environment: interfaces.IEnvironment, side):
        points = environment.get_available_points()
        actions = []
        for i in range(len(points[0])):
            actions.append(wuziqi.WuziqiAction(points[0][i], points[1][i], side))
        return actions

    @staticmethod
    def contain_action(ps, p):
        for item in ps:
            if item.x == p.x and item.y == p.y:
                return True
        return False

    @staticmethod
    def merge_actions(p1, p2):
        return p1 + [p for p in p2 if not CompetingAgent.contain_action(p1, p)]

    def get_candidate_actions(self, environment: interfaces.IEnvironment, last_action):

        def pos_to_action(pos: wuziqi.Position):
            return wuziqi.WuziqiAction(pos.x, pos.y, self.side)

        # select_count = int(self.think_width * self.greedy_rate)

        # if last_action is None:
        #     direct_neighbors_1 = []
        # else:
        #     direct_neighbors_1 = environment.neighbor(wuziqi.Position(last_action.x, last_action.y), 2, True)
        # if environment.last_action is None:
        #     direct_neighbors_2 = []
        # else:
        #     direct_neighbors_2 = environment.neighbor(
        #         wuziqi.Position(environment.last_action.x, environment.last_action.y), 2, True)

        if last_action is None:
            indirect_neighbors_1 = []
        else:
            indirect_neighbors_1 = environment.neighbor(wuziqi.Position(last_action.x, last_action.y), 5,
                                                        True)
        if environment.last_action is None:
            indirect_neighbors_2 = []
        else:
            indirect_neighbors_2 = environment.neighbor(
                wuziqi.Position(environment.last_action.x, environment.last_action.y), 5, True)

        neighbor_actions = [a for a in [pos_to_action(pos) for pos
                                        in self.merge_actions(indirect_neighbors_1, indirect_neighbors_2)]]

        select_count = int(self.search_width/2)

        policy_actions = [a for a in self.merge_actions(self.policy.suggest(environment.get_state(), self.side, select_count),
                                           self.policy.suggest(environment.reverse().get_state(), self.side, select_count))
                          if self.contain_action(neighbor_actions, a)]

        # indirect_neighbor_actions = [a for a in
        #                              [pos_to_action(pos) for pos in merge(indirect_neighbors_1, indirect_neighbors_2)]
        #                              if contains(policy_actions, a) and not contains(direct_neighbor_actions, a)]

        # sample_count = 20 - len(direct_neighbor_actions)
        #
        # if len(no_direct_neighbor_policy_actions_) > sample_count:
        #     actions = direct_neighbor_actions + no_direct_neighbor_policy_actions_[0:sample_count]
        # else:
        #     actions = direct_neighbor_actions + no_direct_neighbor_policy_actions_

        # random_count = self.search_width - len(actions)
        # indirect_neighbor_count = int(random_count * 0.8)
        #
        # if indirect_neighbor_count >= len(indirect_neighbor_actions):
        #     selected_indirect_neighbor_actions = indirect_neighbor_actions
        # elif self.qnet_error > 0.0005:
        #     selected_indirect_neighbor_actions = random.sample(indirect_neighbor_actions, indirect_neighbor_count)
        # else:
        #     selected_indirect_neighbor_actions = self.qnet.suggest(environment, indirect_neighbor_actions, indirect_neighbor_count)
        #
        # actions += selected_indirect_neighbor_actions
        #
        # available_actions = [a for a in self. get_available_actions(environment, self.side) if not contains(actions, a)]
        # random_count -= len(selected_indirect_neighbor_actions)
        # if random_count >= len(available_actions):
        #     random_actions = available_actions
        # elif random_count > 0:
        #     random_actions = random.sample(available_actions, random_count)
        # else:
        #     random_actions = []
        #
        # actions += random_actions

        return policy_actions, policy_actions

    def rehearsal(self, environment: interfaces.IEnvironment, action, opponent_policy: interfaces.IPolicy, steps):
        def get_reward(side):
            return wuziqi.WuziqiGame.eval_state(environment.board_size, environment.get_state()) * side

        def get_partial_random_action(best_actions):
            return game.utils.partial_random(best_actions[0], best_actions[1:], self.greedy_rate)

        def get_possibilities():
            other_possibilities = (1 - self.policy_accuracy[2]) / 10
            top1_possibilities = self.policy_accuracy[0] + other_possibilities
            top5_possibilities = (self.policy_accuracy[1]-self.policy_accuracy[0])/4 + other_possibilities
            top10_possibilities = (self.policy_accuracy[2]-self.policy_accuracy[1])/5 + other_possibilities

            return np.vstack(([[top1_possibilities]],
                       np.ones((4, 1)) * top5_possibilities,
                       np.ones((5, 1)) * top10_possibilities)).reshape((10,))

        def get_action_from_policy(policy: interfaces.IPolicy):
            p = get_possibilities()
            candidates = policy.suggest(environment.get_state(), self.side, 10)
            chosen_action = np.random.choice(candidates, p=p)
            action_val = self.qnet.evaluate(environment.get_state(), chosen_action)
            reversed_state = environment.reverse().get_state()
            reversed_candidates = policy.suggest(reversed_state, self.side, 10)
            chosen_reversed_action = np.random.choice(reversed_candidates, p=p)
            reversed_action_val = self.qnet.evaluate(reversed_state, chosen_reversed_action)
            if action_val > reversed_action_val:
                return chosen_action
            else:
                return chosen_reversed_action

        history1 = []
        history2 = []
        next_state_1 = environment.get_state().copy()
        next_action_1 = action
        final_state = 0
        next_action_2 = None

        while steps > 0:
            steps -= 1
            environment.update(next_action_1)
            r1 = get_reward(self.side)
            history1.append([next_state_1, next_action_1, r1])
            if environment.is_ended():
                next_state_1 = environment.get_state().copy()
                next_action_1 = wuziqi.WuziqiAction(0, 0, 0)
                history1.append([next_state_1, next_action_1, 0])
                if len(history2) > 0:
                    # history2[-1][-1] = -1
                    history2.append([next_state_1 * -1, wuziqi.WuziqiAction(0, 0, 0), 0])
                final_state = 1
                break
            else:
                reversed_environment = environment.reverse()
                next_state_2 = reversed_environment.get_state().copy()
                # _, candidate_actions = self.get_candidate_actions(reversed_environment, next_action_2)
                # next_action_2 = self.qnet.suggest(reversed_environment, candidate_actions, 1)[0]
                next_action_2 = get_action_from_policy(opponent_policy)

                state = environment.update(next_action_2.reverse()).copy()
                r2 = get_reward(self.side * -1)
                history2.append([next_state_2, next_action_2, r2])

                if environment.is_ended():
                    # history1[-1][-1] = -1
                    next_action_1 = wuziqi.WuziqiAction(0, 0, 0)
                    history1.append([state, next_action_1, 0])
                    history2.append([state * -1, next_action_1, 0])
                    final_state = -1
                    break
                else:
                    next_state_1 = state
                    # _, candidate_actions = self.get_candidate_actions(environment, next_action_1)
                    # next_action_1 = self.qnet.suggest(environment, candidate_actions, 1)[0]

                    next_action_1 = get_action_from_policy(self.policy)
        return history1, history2, final_state

    def save(self, save_dir):
        self.qnet.save(save_dir)
        self.policy.save(save_dir)

    def restore(self, restore_dir):
        self.qnet.restore(restore_dir)
        self.policy.restore(restore_dir)

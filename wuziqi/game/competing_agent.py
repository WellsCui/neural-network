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
        self.minimum_learning_rate = 0.0004
        self.learning_rate_dacade_rate = 0.6
        self.policy = game.wuziqi_policy_net.WuziqiPolicyNet(name + "_", board_size, initial_learning_rate, lbd)
        self.qnet = game.wuziqi_value_net.WuziqiQValueNet(name + "_", board_size, initial_learning_rate, lbd)
        self.mode = "online_learning."
        self.lbd = lbd
        self.search_depth = 20
        self.search_width = 10
        self.policy_training_data = []
        self.value_net_training_data = []
        self.value_net_training_size = 50
        self.policy_training_size = 50
        self.epsilon = 0.001
        self.greedy_rate = 0.5
        self.board_size = board_size
        self.is_greedy = False
        self.last_action = None
        self.qnet_error = 1
        self.policy_accuracy = [0, 0, 0]
        self.online_learning = True

    def act(self, environment: interfaces.IEnvironment):
        state = environment.get_state().copy()
        policy_actions, candidates = self.get_candidate_actions(environment, self.policy, self.last_action)
        rehearsals = []
        reversed_rehearsals = []
        reversed_environment = environment.reverse()

        for act in candidates:
            rehearsals.append(self.rehearsal(environment.clone(), act, self.policy, self.search_depth))
            reversed_rehearsals.append(self.rehearsal(reversed_environment.clone(), act, self.policy, self.search_depth))

        best_action = self.choose_best_action_from_rehearsals(rehearsals, reversed_rehearsals)

        if self.qnet_error < 0.0005:
            self.policy_training_data.append([state, best_action])

        if self.is_greedy:
            action = best_action
        else:
            action = game.utils.partial_random(best_action, candidates, self.greedy_rate)

        print("%s: best (%d, %d): %f, in policy actions: %d, %s,  actual (%d, %d): %f best: %s" %
              (self.name,
               best_action.x, best_action.y,
               self.qnet.evaluate(state, best_action),
               len(policy_actions),
               self.contain_action(policy_actions, best_action),
               action.x, action.y,
               self.qnet.evaluate(state, action),
               action == best_action))

        if self.online_learning:
            self.learn_from_rehearsals(rehearsals + reversed_rehearsals)
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

    def learn_value_net_from_session(self, session):
        history, opponent_history, final_state = session
        if self.qnet_error > 0.001:
            if final_state == 1:
                self.value_net_training_data.append(history)
            elif final_state == -1:
                self.value_net_training_data.append(opponent_history)
        else:
            self.value_net_training_data.append(history)
            self.value_net_training_data.append(opponent_history)

        if len(self.value_net_training_data) >= self.value_net_training_size:
            error = self.qnet.train(self.value_net_learning_rate, self.value_net_training_data)
            if error > 0 and self.value_net_learning_rate > self.minimum_learning_rate:
                self.value_net_learning_rate *= self.learning_rate_dacade_rate
                self.qnet_error = error
            self.value_net_training_data = []

    def learn_policy_net_from_session(self, session, learn_from_opponent=False):
        history, opponent_history, final_state = session
        if final_state == 1:
            winning_state, winning_action, _ = history[-2]
            self.policy_training_data.append([winning_state, winning_action])
            if len(opponent_history) > 1:
                failing_state, failing_action, _ = opponent_history[-2]
                self.policy_training_data.append([failing_state, winning_action])
        elif final_state == -1:
            if learn_from_opponent:
                for s, a, r in opponent_history[:-1]:
                    self.policy_training_data.append([s, a])
            else:
                winning_state, winning_action, _ = opponent_history[-2]
                self.policy_training_data.append([winning_state, winning_action])
                failing_state, failing_action, _ = history[-2]
                self.policy_training_data.append([failing_state, winning_action])

        if len(self.policy_training_data) > self.policy_training_size:
            self.policy_accuracy = self.policy.train(self.policy_net_learning_rate, self.policy_training_data)
            if self.policy_net_learning_rate > self.minimum_learning_rate:
                self.policy_net_learning_rate *= self.learning_rate_dacade_rate
            self.policy_training_data = []

    def learn_from_session(self, session, learn_from_opponent=False):
        self.learn_value_net_from_session(session)
        self.learn_policy_net_from_session(session, learn_from_opponent)

    def learn_from_rehearsals(self, rehearsals):
        for session in rehearsals:
            self.learn_from_session(session)

    def evaluate_rehearsal(self, rehearsal):
        history, opponent_history, final_state = rehearsal
        state, action, reward = history[-1]

        if final_state == 0:
            result = self.qnet.evaluate(state, action)
            if len(opponent_history) > 0:
                opponent_state, opponent_action, opponent_reward = opponent_history[-1]
                result -= self.qnet.evaluate(opponent_state, opponent_action)
            result *= (self.lbd ** (len(history) - 1))
        else:
            result = final_state * (self.lbd ** (len(history) - 2))
        print("action (%d, %d) len %d rehearsal val: %f , final state: %d" %
              (history[0][1].x, history[0][1].y, len(history), result, final_state))
        action_str = []
        for s, a, r in history:
            action_str.append('(%d, %d)' % (a.x, a.y))
        print(' '.join(action_str))
        action_str = []
        for s, a, r in opponent_history:
            action_str.append('(%d, %d)' % (a.x, a.y))
        print(' '.join(action_str))


        return result

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

    def get_neighbor_actions(self, environment: interfaces.IEnvironment, last_action):
        def pos_to_action(pos: wuziqi.Position):
            return wuziqi.WuziqiAction(pos.x, pos.y, self.side)

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
        return neighbor_actions

    def get_candidate_actions(self, environment: interfaces.IEnvironment, policy, last_action):

        neighbor_actions = self.get_neighbor_actions(environment, last_action)

        select_count = self.search_width

        policy_actions = [a for a in
                          self.merge_actions(policy.suggest(environment.get_state(), self.side, select_count),
                                             policy.suggest(environment.reverse().get_state(), self.side, select_count))
                          if self.contain_action(neighbor_actions, a)]

        if len(policy_actions) >= self.search_width:
            policy_actions = np.random.choice(policy_actions, self.search_width)
            return policy_actions, policy_actions
        else:
            random_count = self.search_width - len(policy_actions)
            random_actions = np.ndarray.tolist(np.random.choice([a for a in neighbor_actions
                                                                 if not self.contain_action(policy_actions, a)],
                                                                random_count))
            return policy_actions, policy_actions + random_actions

    def rehearsal(self, environment: interfaces.IEnvironment, action, opponent_policy: interfaces.IPolicy, steps):
        def get_reward(side):
            return environment.eval_state() * side

        def get_partial_random_action(best_actions):
            if len(best_actions) == 1:
                return best_actions[0]
            else:
                return game.utils.partial_random(best_actions[0], best_actions[1:], self.greedy_rate)

        def get_possibilities():
            other_possibilities = (1 - self.policy_accuracy[2]) / 10
            top1_possibilities = self.policy_accuracy[0] + other_possibilities
            top5_possibilities = (self.policy_accuracy[1]-self.policy_accuracy[0])/4 + other_possibilities
            top10_possibilities = (self.policy_accuracy[2]-self.policy_accuracy[1])/5 + other_possibilities

            return np.vstack(([[top1_possibilities]],
                       np.ones((4, 1)) * top5_possibilities,
                       np.ones((5, 1)) * top10_possibilities)).reshape((10,))

        def get_action_from_policy(policy, environment, last_action):
            neighbor_actions = self.get_neighbor_actions(environment, last_action)
            if self.is_greedy:
                policy_actions = self.merge_actions(
                    policy.suggest(environment.get_state(), self.side, self.search_width),
                    policy.suggest(environment.get_state()*-1, self.side, self.search_width)
                )

                candidate_action = [a for a in policy_actions if self.contain_action(neighbor_actions, a)]
                if len(candidate_action) == 0:
                    candidate_action = neighbor_actions
                return self.qnet.suggest(environment, candidate_action, 1)[0]
            else:
                policy_actions = self.merge_actions(
                    policy.suggest(environment.get_state(), self.side, self.search_width),
                    policy.suggest(environment.get_state()*-1, self.side, self.search_width)
                )
                if len(neighbor_actions) == 0:
                    return np.random.choice(policy_actions)
                elif np.random.choice(2, p=[self.greedy_rate, 1 - self.greedy_rate]) == 0:
                    return np.random.choice(policy_actions)
                else:
                    return np.random.choice(neighbor_actions)

        history1 = []
        history2 = []
        next_state_1 = environment.get_state().copy()
        next_action_1 = action
        final_state = 0
        next_action_2 = environment.last_action.reverse()

        while steps > 0:
            steps -= 1
            environment.update(next_action_1)
            r1 = get_reward(self.side)
            history1.append([next_state_1, next_action_1, r1])
            if environment.is_ended():
                if r1 == 1:
                    next_state_1 = environment.get_state().copy()
                    next_action_1 = wuziqi.WuziqiAction(0, 0, 0)
                    history1.append([next_state_1, next_action_1, 0])
                    if len(history2) > 0:
                        history2.append([next_state_1 * -1, wuziqi.WuziqiAction(0, 0, 0), 0])
                    final_state = 1
                break
            else:
                reversed_environment = environment.reverse()
                next_state_2 = reversed_environment.get_state().copy()
                next_action_2 = get_action_from_policy(opponent_policy, reversed_environment, next_action_2)

                state = environment.update(next_action_2.reverse()).copy()
                r2 = get_reward(self.side * -1)
                history2.append([next_state_2, next_action_2, r2])

                if environment.is_ended():
                    if r2 == 1:
                        next_action_1 = wuziqi.WuziqiAction(0, 0, 0)
                        history1.append([state, next_action_1, 0])
                        history2.append([state * -1, next_action_1, 0])
                        final_state = -1
                    break
                else:
                    next_state_1 = state
                    next_action_1 = get_action_from_policy(self.policy, environment, next_action_1)
        return history1, history2, final_state

    def save(self, save_dir):
        self.qnet.save(save_dir)
        self.policy.save(save_dir)

    def restore(self, restore_dir):
        self.qnet.restore(restore_dir)
        self.policy.restore(restore_dir)

    def learn_from_experience(self, experience, learn_from_opponent):
        def reverse_history(h):
            return [[state * -1, action.reverse(), reward] for state, action, reward in h]
        history, opponent_history, final_result = experience
        session = history, reverse_history(opponent_history), final_result
        self.learn_from_session(session, learn_from_opponent)

    def train_model_with_raw_data(self, train_dir):
        print('training model with raw data...')
        self.policy.training_data_dir = train_dir
        self.qnet.training_data_dir = train_dir
        self.policy.train_with_file()
        self.qnet.train_with_file()

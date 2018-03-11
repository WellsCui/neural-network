import datetime

import game.wuziqi as wuziqi
import numpy as np
import math
import threading
import concurrent
import concurrent.futures as futures
import game.alpha_zero.queued_evaluator as queued_evaluator
import game.synchronizable_object as synchronizable_object


class McNode(synchronizable_object.SynchronizableObject):
    def __init__(self, state, side):
        synchronizable_object.SynchronizableObject.__init__(self)
        self.state = state
        self.edges = []
        self.value = 0
        self.expended = False
        self.side = side
        self.visit_count = 0
        self.mask = np.reshape(state == 0, self.state.shape[0] * self.state.shape[1])


class McEdge(synchronizable_object.SynchronizableObject):
    def __init__(self, src_node: McNode, action: wuziqi.WuziqiAction, dst_node: McNode, probability):
        synchronizable_object.SynchronizableObject.__init__(self)
        self.src_node = src_node
        self.action = action
        self.dst_node = dst_node
        self.probability = probability
        self.action_value = 0.0
        self.total_action_value = 0.0
        self.visit_count = 0
        self.upper_confidence_bound = 0.0


class McTreeSearchOption:
    def __init__(self, simulations, temperature, c_put):
        self.simulations = simulations
        self.temperature = temperature
        self.c_put = c_put
        self.evalue_batch_size = 8
        self.max_simulations = 50
        self.score_lambda = 0.99
        self.thether = 0.25


class McTreeSearch:
    def __init__(self, root: McNode, net, evaluator: queued_evaluator.QueuedEvaluator, options: McTreeSearchOption):
        self.root = root
        self.net = net
        self.evaluator = evaluator
        self.nodes = [self.root]
        self.options = options
        self.positions = [a for a in np.ndindex(self.net.board_size)]
        self.position_count = len(self.positions)

    def get_upper_confidence_bound(self, edge: McEdge):
        return self.options.c_put * edge.probability * math.sqrt(edge.src_node.visit_count) * (1 + edge.visit_count)

    def get_dst_node(self, edge: McEdge):
        if edge.dst_node:
            return edge.dst_node
        if edge.action is None:
            return None
        dst_state = edge.src_node.state.copy()
        dst_state[edge.action.y, edge.action.x] = edge.action.val
        node = self.get_node_by_state(dst_state)
        if node:
            return node
        node = McNode(dst_state, edge.action.val)
        self.nodes.append(node)
        return node

    def create_edge(self, src_node, action, probability):
        return McEdge(src_node, action, None, probability)

    def create_edge_v(self, node, pos, p):
        action = wuziqi.WuziqiAction(pos[1], pos[0], node.side * -1)
        return self.create_edge(node, action, p)

    def build_edges(self, node, probabilities):
        edges = []
        for i in range(self.position_count):
            if not node.mask[i]:
                continue
            else:
                edges.append(self.create_edge_v(node, self.positions[i], probabilities[i]))
        return edges

    def expand_node(self, node: McNode):
        if node.expended:
            return node
        ft1 = futures.Future()
        self.evaluator.submit_request(node.state, ft1)
        rotation = np.random.choice(4, 1)[0]

        ft2 = futures.Future()
        self.evaluator.submit_request(np.rot90(node.state, rotation), ft2)

        p, v = ft1.result()
        p_, v_ = ft2.result()
        p_ = np.rot90(p_.reshape(self.net.board_size), rotation).reshape(self.net.board_size[0]*self.net.board_size[1])

        p = p * (1 - self.options.thether) + p_ * self.options.thether

        def update_node():
            node.value = v[0]
            if not wuziqi.WuziqiGame(self.net.board_size, node.state).is_ended():
                node.edges = self.build_edges(node, p)
            node.expended = True

        node.synchronize(update_node)
        return node

    def update_upper_confidence_bound(self, node: McNode):
        def update_node():
            node.visit_count += 1

        node.synchronize(update_node)
        node_visit_count_sqrt = math.sqrt(node.visit_count)
        for edge in node.edges:
            def update_edge():
                edge.upper_confidence_bound = self.options.c_put * edge.probability * \
                                              node_visit_count_sqrt * (1 + edge.visit_count)

            edge.synchronize(update_edge)


    def back_up(self, steps, val):

        for step in reversed(steps):
            # step.src_node.visit_count += 1
            step.visit_count += 1
            step.total_action_value += val
            step.action_value = step.total_action_value / step.visit_count
            self.update_upper_confidence_bound(step.src_node)

    def select_edge(self, node: McNode):

        if len(node.edges) == 0:
            return None
        rs = [-1, None]
        for edge in node.edges:
            # edge_puct = edge.action_value + edge.upper_confidence_bound
            edge_puct = edge.action_value + self.get_upper_confidence_bound(edge)

            if edge_puct > rs[0]:
                rs = [edge_puct, edge]
        return rs[1]

    def simulate(self, simulate_index):
        current = self.root
        steps = []
        while current is not None and current.expended:
            step = self.select_edge(current)
            if step is None:
                break
            steps.append(step)
            current = self.get_dst_node(step)
        if current is not None and not current.expended:
            self.expand_node(current)
        self.back_up(steps, current.value)

    def collect_node_statistics(self, node):
        p = np.array([math.pow(edge.visit_count, self.options.temperature) for edge in node.edges])
        p_sum = np.sum(p)
        return p / p_sum

    def set_root(self, node: McNode):
        self.root = node
        self.nodes = self.get_nodes(self.root)

    def build_node_probabilities(self, node, statistics):
        p = np.zeros(self.net.board_size)
        for i in range(len(node.edges)):
            p[node.edges[i].action.x, node.edges[i].action.y] = statistics[i]
        return p.reshape(self.net.board_size[0] * self.net.board_size[1])

    def run_simulations(self):
        begin = datetime.datetime.now()
        print('running simulations...')
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.simulate, simulate_index): simulate_index
                       for simulate_index in range(self.options.simulations)}
            for future in concurrent.futures.as_completed(futures):
                simulate_index = futures[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('simulate %s threw an exception: %s' % (simulate_index, exc))
                # else:
                #     print('simulate %s return: %s' % (simulate_index, data))

        end = datetime.datetime.now()
        print('simulations completed in %d seconds:' % (end-begin).seconds)

    def move(self):
        self.run_simulations()
        statistics = self.collect_node_statistics(self.root)
        edge = self.root.edges[np.argmax(statistics)]
        probabilities = self.build_node_probabilities(edge.src_node, statistics)
        print('selected action: (%d, %d) side: %d, visit_count: %d, action_value: %f' %
              (edge.action.x, edge.action.y, edge.action.val, edge.visit_count,  edge.action_value))
        print('src_node.visit_count: %d, src_node.value: %f' % (edge.src_node.visit_count, edge.src_node.value))
        self.set_root(self.get_dst_node(edge))
        wuziqi.WuziqiGame(self.net.board_size, self.root.state, edge.action).show()
        return edge.src_node.state, probabilities

    def self_play(self):
        steps = []
        c = 0
        while not wuziqi.WuziqiGame(self.net.board_size, self.root.state).is_ended():
            if c > 10 and self.options.temperature == 1:
                self.options.temperature = 1e-8
            state, p = self.move()
            steps.append((state, p))
            c += 1
            print('step:', c)

        return self.score_steps(steps)

    def score_steps(self, steps):
        val = wuziqi.WuziqiGame(self.net.board_size, self.root.state).eval_state()
        rs = []
        steps.reverse()
        for s, p in steps:
            rs.append((s, p, val))
            val *= self.options.score_lambda
        rs.reverse()
        return rs

    def get_node_by_state(self, state):
        for node in self.nodes:
            if wuziqi.WuziqiGame.is_same_state(node.state, state):
                return node
        return None

    def get_nodes(self, node: McNode):
        result = [node]
        for edge in node.edges:
            if edge.dst_node:
                result += self.get_nodes(edge.dst_node)
        return result

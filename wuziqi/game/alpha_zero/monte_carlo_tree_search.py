import datetime

import game.wuziqi as wuziqi
import numpy as np
import math
import threading

class McNode:
    def __init__(self, state, side):
        self.state = state
        self.edges = []
        self.value = 0
        self.expended = False
        self.side = side
        self.visit_count = 0


class McEdge:
    def __init__(self, src_node: McNode, action, dst_node: McNode, probability):
        self.src_node = src_node
        self.action = action
        self.dst_node = dst_node
        self.probability = probability
        self.action_value = 0.0
        self.total_action_value = 0.0
        self.visit_count = 0


class McTreeSearchOption:
    def __init__(self, simulations, temperature, c_put):
        self.simulations = simulations
        self.temperature = temperature
        self.c_put = c_put
        self.evalue_batch_size = 8


class McTreeSearch:
    def __init__(self, root: McNode, net, options: McTreeSearchOption):
        self.root = root
        self.net = net
        self.nodes = [self.root]
        self.options = options

    def get_upper_confidence_bound(self, edge: McEdge):
        return self.options.c_put * edge.probability * math.sqrt(edge.src_node.visit_count) * (1 + edge.visit_count)

    def get_dst_node(self, src_node, action):
        if action is None:
            return None
        dst_state = src_node.state.copy()
        dst_state[action.y, action.x] = action.val
        dst_node = self.get_node_by_state(dst_state)
        if dst_node:
            return dst_node
        dst_node = McNode(dst_state, action.val)
        self.nodes.append(dst_node)
        return dst_node

    def create_edge(self, src_node, action, probability):
        return McEdge(src_node, action, self.get_dst_node(src_node, action), probability)

    def build_edges(self, node, probabilities):
        mask = np.reshape(node.state == 0, self.net.board_size[0] * self.net.board_size[1])
        action_probabilities = probabilities * mask
        positions = [a for a in np.ndindex(self.net.board_size)]



        def create_edge_v(pos, p):
            action = wuziqi.WuziqiAction(pos[1], pos[0], node.side * -1)
            return self.create_edge(node, action, p)

        # create_edge_v_func = np.vectorize(create_edge_v)
        # edges = create_edge_v_func(actions, action_probabilities)

        edges = []

        for i in range(len(positions)):
            if not mask[i]:
                continue
            else:
                edges.append(create_edge_v(positions[i], action_probabilities[i]))
        return edges

    def expand_node(self, node: McNode):
        if node.expended:
            return node
        p, v = self.net.evaluate(node.state)
        node.value = v[0][0]
        if not wuziqi.WuziqiGame(self.net.board_size, node.state).is_ended():
            node.edges = self.build_edges(node, p[0])
        node.expended = True
        return node

    def back_up(self, steps, val):
        # print('back up val:', val)
        for step in reversed(steps):

            step.src_node.visit_count += 1
            step.visit_count += 1
            step.total_action_value += val
            step.action_value = step.total_action_value / step.visit_count
            # print('step: (%s,%s) step.src_node.visit_count: %d, step.visit_count: %d, step.total_action_value: %f , step.action_value: %f'
            #       % (step.action.x, step.action.y, step.src_node.visit_count, step.visit_count, step.total_action_value, step.action_value))

    def select_action(self, node: McNode):

        if len(node.edges) == 0:
            return None

        rs = [-1, None]
        for edge in node.edges:
            edge_puct = edge.action_value + self.get_upper_confidence_bound(edge)
            # print('action_value: %s' % (edge.action_value, edge.visit_count))
            if edge_puct > rs[0]:
                rs = [edge_puct, edge]
        return rs[1]

    def simulate(self):
        current = self.root
        steps = []
        # print('current.expended:', current.expended)
        while current is not None and current.expended:
            step = self.select_action(current)
            # print('step:', step.action)
            if step is None:
                break

            steps.append(step)
            current = step.dst_node
        if current is not None and not current.expended:
            # print('expanding node:', current)
            self.expand_node(current)
        self.back_up(steps, current.value)

    def collect_node_statistics(self, node):
        p = np.array([math.pow(edge.visit_count, self.options.temperature) for edge in node.edges])
        p_sum = np.sum(p)
        return p / p_sum

    def execute_edge(self, edge: McEdge):
        self.root = edge.dst_node

    def build_node_probabilities(self, node, statistics):
        p = np.zeros(self.net.board_size)
        for i in range(len(node.edges)):
            p[node.edges[i].action.x, node.edges[i].action.y] = statistics[i]
        return p.reshape(self.net.board_size[0] * self.net.board_size[1])

    def run_simulations(self):
        begin = datetime.datetime.now()
        threads = []
        for i in range(self.options.simulations):
            t = threading.Thread(target=self.simulate)
            print('running simulation:', i)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        end = datetime.datetime.now()
        print('simpulations time:', (end-begin).seconds)


    def play(self):
        self.run_simulations()
        statistics = self.collect_node_statistics(self.root)
        edge = self.root.edges[np.argmax(statistics)]
        statistics = self.build_node_probabilities(edge.src_node, statistics).reshape(self.net.board_size)
        print('selected action: (%d, %d) side: %d, action_value: %f' % (edge.action.x, edge.action.y, edge.action.val, edge.action_value))
        print('src_node.visit_count: %d, src_node.value: %f' % (edge.src_node.visit_count, edge.src_node.value))
        # print('statistics:')
        # print(statistics)
        self.execute_edge(edge)
        wuziqi.WuziqiGame(self.net.board_size, self.root.state, edge.action).show()
        return edge.src_node.state, statistics

    def get_node_by_state(self, state):
        for node in self.nodes:
            if wuziqi.WuziqiGame.is_same_state(node.state, state):
                return node
        return None
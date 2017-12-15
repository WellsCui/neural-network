import numpy as np
import time
import logging
import os
import sys
import game.wuziqi as wuziqi
import game.agents as agents
import game.evaluators as evaluators
import game.actor_critic_agent as ac_agent
import game.competing_agent as competing_agent
import game.value_policy_agent as value_policy_agent
import game.human_agent as human_agent
import game.interfaces as interfaces
import game.alpha_zero.training as alpha_training
import training.bdt_game
import training.psq_game
import tables
import game.utils
import re


def play(times):
    player1 = agents.WuziqiRandomAgent(1)
    player2 = agents.WuziqiRandomAgent(-1)

    for i in range(times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame((11, 11))
        step = 0
        sleep_time = 0
        while True:
            step += 1
            action = player1.act(game)
            print("player1 toke action:", action.x, action.y)
            game.show()
            time.sleep(sleep_time)
            if game.is_ended():
                print("game ended after player1 toke action:", action.x, action.y)
                break

            action = player2.act(game)
            print("player2 toke action:", action.x, action.y)
            game.show()
            time.sleep(sleep_time)
            if game.is_ended():
                print("game ended after player2 toke action:", action.x, action.y)
                break
        print("Game is ended on step:", step)
        val = game.eval_state()
        if val == 1:
            winner = "player1"
        elif val == -1:
            winner = "player2"
        else:
            winner = "nobody"
        print(" winner is", winner)

        # game.show()


# play(10)


def train_evaluator(game_times, validate_times):
    player1 = agents.WuziqiRandomAgent(1)
    player2 = agents.WuziqiRandomAgent(-1)
    game_size = (15, 15)
    evaluator = evaluators.WuziqiEvaluator(0.95, game_size, 100, 0.001, 1)
    games = []
    for i in range(game_times+validate_times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame(game_size)
        states = [np.copy(game.get_state())]
        step = 0
        while True:
            step += 1
            action = player1.act(game)
            states.append(np.copy(game.get_state()))
            # print(game.get_state())
            # print("player1 take action:", action.x, action.y)
            if game.is_ended():
                break
            action = player2.act(game)
            states.append(np.copy(game.get_state()))
            # print(game.get_state())
            # print("player2 take action:", action.x, action.y)
            if game.is_ended():
                break
        if i < game_times:
            games.append(states)
            if i % 10 == 0:
                print("training with 10 games ...")
                evaluator.train(games)
                games.clear()
            if i % 20 == 0 or i % 23 == 0:
                val = game.eval_state()
                if val == 1:
                    winner = "player1"
                elif val == -1:
                    winner = "player2"
                else:
                    winner = "nobody"
                preds = evaluator.predict(states[-2:])
                print(" winner is", winner, "predict without training:", preds)

        else:
            val = game.eval_state()
            if val == 1:
                winner = "player1"
            elif val == -1:
                winner = "player2"
            else:
                winner = "nobody"
            preds = evaluator.predict(states[-2:])
            print(" winner is", winner, "predict without training:", preds)


        # print(states)
        # print("Game is ended on step:", step)

        # print("state count:", len(states))


        # if i % validate_frequency > 0:
        #     evaluator.train(states)
        #     # print(" winner is", winner, "predict after training :", evaluator.predict(states[-2:]))
        # else:
        #     preds = evaluator.predict(states[-2:])
        #     print(" winner is", winner, "predict without training:", preds)
        # game.show()


def get_reward(game):
    return game.eval_state()


def run_actor_critic_agent(times):
    player1 = ac_agent.ActorCriticAgent((11, 11), 0.001, 1, 0.95)
    player2 = agents.WuziqiRandomAgent(-1)

    # q_net = ac_agent.WuziqiQValueNet((11, 11), 0.001, 1, 0.95)
    # q_net.build_state_action(game.get_state(), wuziqi.WuziqiAction(1, 1, 1))
    wins = 0
    for i in range(times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame((11, 11))
        step = 0
        current_state = game.get_state()
        current_action = player1.act(game)

        while True:
            step += 1
            r = get_reward(game)
            if game.is_ended():
                next_state = game.get_state()
                next_action = wuziqi.WuziqiAction(0, 0, 0)
                player1.learn(current_state, current_action, r, next_state, next_action)
                break
            else:
                player2.act(game)
                next_state = game.get_state()
                if game.is_ended():
                    game.show()
                    next_action = wuziqi.WuziqiAction(0, 0, 0)
                    player1.learn(current_state, current_action, get_reward(game), next_state, next_action)
                    break
                else:
                    next_action = player1.act(game)
            game.show()
            player1.learn(current_state, current_action, r, next_state, next_action)
            current_action = next_action
            current_state = next_state

        print("Game is ended on step:", step)
        val = game.eval_state()
        if val == 1:
            wins += 1
            winner = "player1 " + str(wins) + " out of " + str(i+1)

        elif val == -1:
            winner = "player2 " + str(i+1 - wins) + " out of " + str(i+1)
        else:
            winner = "nobody"
        print(" winner is", winner)
        # game.show()


def reverse_history(history):
    return [[state * -1, action.reverse(), reward] for state, action, reward in history]


def learn_from_experience(player: competing_agent.CompetingAgent, history, opponent_history, final_result):
    session = history, reverse_history(opponent_history), final_result
    player.learn_value_net_from_sessions([session])


def run_with_agents(times, player1: interfaces.IAgent, player2: interfaces.IAgent,
                    model1_path, model2_path, online_learning=True, learn_from_winner=True, is_save_game=True):
    moves = []
    hex_str = '0123456789ABCDEF'

    def move(player, game):
        state = game.get_state().copy()
        action = player.act(game)
        moves.append(hex_str[action.x] + hex_str[action.y])
        is_ended = game.is_ended()
        if is_ended:
            reward = game.eval_state() * player.side
        else:
            reward = 0
        if player == player1:
            history1.append([state, action, reward])
        else:
            history2.append([state, action, reward])
        if is_ended:
            end_state = game.get_state().copy()
            history1.append([end_state, wuziqi.WuziqiAction(0, 0, 0), 0])
            history2.append([end_state, wuziqi.WuziqiAction(0, 0, 0), 0])

        game.show()
        return is_ended

    board_size = player1.board_size
    wins = 0

    for i in range(times):
        print("Starting Game ", i)
        game = wuziqi.WuziqiGame(board_size)
        moves = []
        history1 = []
        history2 = []
        step = 0
        if i % 2 == 0:
            first_player = player1
            second_player = player2
        else:
            first_player = player2
            second_player = player1
        # player1.is_greedy = i % 11 == 0
        # player2.is_greedy = i % 13 == 0

        # print("player1 greedy: ", player1.is_greedy)
        # print("player2 greedy: ", player2.is_greedy)
        while True:
            step += 1
            if move(first_player, game):
                break
            if move(second_player, game):
                break

        print("Game is ended on step:", step)
        val = game.eval_state()
        if is_save_game:
            save_game(moves, 'data/games.bdt')

        if online_learning:
            player1.learn_from_experience([history1, history2, player1.side * val], learn_from_winner)
            player2.learn_from_experience([history2, history1, player2.side * val], learn_from_winner)

        if val == 1:
            wins += 1
            winner = "player1 " + str(wins) + " out of " + str(i+1)
            if player1 is competing_agent.CompetingAgent:
                player1.increase_greedy()
        elif val == -1:
            winner = "player2 " + str(i+1 - wins) + " out of " + str(i+1)
            if player2 is competing_agent.CompetingAgent:
                player2.increase_greedy()
        else:
            winner = "nobody"
        print("Winner is", winner)
        if online_learning:
            print("Saving player1 model...")
            player1.save_model(model1_path)
            player2.save_model(model2_path)


def save_game(moves, bdt_file):
    game = '%s=[%s,%s,%s,%s,?,?]\n' % ('2017,1', 'player1', 'player2', '+', ''.join(moves))
    with open(bdt_file, 'a') as f:
        f.write(game)


def ai_vs_ai(base_path, save_model):
    board_size = (15, 15)
    training_data_dir=os.path.join(base_path+"/data")
    player1 = competing_agent.CompetingAgent("player1", board_size, 0.0005, 1, 0.99, training_data_dir)
    player2 = competing_agent.CompetingAgent("player2", board_size, 0.0005, -1, 0.99, training_data_dir)
    model1_path = os.path.join(base_path, "model/player1")
    model2_path = os.path.join(base_path, "model/player1")
    if os.path.isfile(os.path.join(model1_path, 'checkpoint')):
        print("Loading player1 model...")
        player1.load_model(model1_path)
    if os.path.isfile(os.path.join(model2_path, 'checkpoint')):
        print("Loading player2 model...")
        player2.load_model(model2_path)

    # player1.train_model_with_raw_data(training_data_dir)
    # player1.save(model_path)
    # train_agent_with_games(player1, model_path)
    player1.online_learning = True
    player2.online_learning = True
    player1.is_greedy = True
    player1.is_greedy = True
    player2.is_greedy = True
    run_with_agents(10, player1, player2, model1_path, model2_path, save_model, False)


def set_logging(level):
    class Unbuffered(object):
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            self.stream.flush()

        def writelines(self, datas):
            self.stream.writelines(datas)
            self.stream.flush()

        def __getattr__(self, attr):
            return getattr(self.stream, attr)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    logging.root.setLevel(level)
    # logging.root.setLevel('DEBUG')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ch.setFormatter(formatter)
    logging.root.addHandler(ch)



def ai_vs_human(base_path, load_from_model=True, online_learning=True, train_with_raw_data=False):

    board_size = (15, 15)
    training_data_dir=os.path.join(base_path+"/data")
    player1 = competing_agent.CompetingAgent("player1", board_size, 0.0005, 1, 0.99, training_data_dir)

    human_player = human_agent.HumanAgent("human", board_size, -1)
    model_path = os.path.join(base_path+"/model")
    if load_from_model and os.path.isfile(os.path.join(model_path+'/checkpoint')):
        print("Loading player1 model...")
        player1.load_model(model_path)
    if train_with_raw_data:
        player1.train_model_with_raw_data(training_data_dir, model_path, 100)
        player1.save_model(model_path)
    # train_agent_with_games(player1, model_path)
    player1.online_learning = online_learning
    player1.is_greedy = True
    run_with_agents(10, player1, human_player, model_path, model_path, online_learning, True)


def ai2_vs_human(base_path, load_from_model=True, online_learning=True, train_with_raw_data=False):
    board_size = (15, 15)
    training_data_dir = os.path.join(base_path + "/data")
    player1 = value_policy_agent.ValuePolicyAgent("player1", board_size, 0.0005, 1, 0.99, training_data_dir)

    human_player = human_agent.HumanAgent("human", board_size, -1)
    model_path = os.path.join(base_path + "/model")
    if load_from_model and os.path.isfile(os.path.join(model_path + '/checkpoint')):
        print("Loading player1 model...")
        player1.load_model(model_path)
    if train_with_raw_data:
        player1.train_model_with_raw_data(training_data_dir, model_path, 100)
        player1.save_model(model_path)
    # train_agent_with_games(player1, model_path)
    player1.online_learning = online_learning
    player1.is_greedy = True
    run_with_agents(10, player1, human_player, model_path, model_path, online_learning, True)


def train_with_games(base_dir, games_file, is_value_policy_agent=False, load_from_model=True,
                     prompt=False):
    training_data_dir = base_dir + '/data'
    model_dir = base_dir + '/model'
    board_size = (15, 15)
    if is_value_policy_agent:
        player1 = value_policy_agent.ValuePolicyAgent("player1", board_size, 0.0005, 1, 0.99)
    else:
        player1 = competing_agent.CompetingAgent("player1", board_size, 0.0005, 1, 0.99)
    if load_from_model and os.path.isfile(os.path.join(model_dir + '/checkpoint')):
        print("Loading player1 model...")
        player1.load_model(model_dir)
    player1.qnet.training_data_dir = training_data_dir
    player1.policy.training_data_dir = training_data_dir
    player1.qnet.model_path = model_dir
    player1.policy.model_path = model_dir
    train_agent_with_games(player1, training.bdt_game.get_sessions(games_file), prompt)


def train_with_raw_data(training_data_dir, model_dir):
    board_size = (15, 15)
    player1 = competing_agent.CompetingAgent("player1", board_size, 0.0005, 1, 0.99)
    # player1.load_model(model_dir)
    player1.qnet.training_data_dir = training_data_dir
    player1.policy.training_data_dir = training_data_dir
    player1.train_model_with_raw_data(training_data_dir, model_dir, 300)
    player1.save_model(model_dir)


def train_agent_with_games(agent, sessions, prompt=False):
    finished_sessions = list(s for s in sessions if s[2] != 0)
    session_count = len(finished_sessions)
    training_num = int(session_count*0.8)
    train_sessions = np.array(finished_sessions[0:training_num])
    batch_size = int(session_count*0.2)
    for i in range(100):
        print("iteration: ", i)
        if prompt:
            learning_rate, epics = eval(input('Enter (learning_rate, epics):'))
        else:
            learning_rate, epics = 1, 50
        agent.qnet.training_epics = epics
        agent.policy.training_epics = epics
        agent.learn_from_sessions(train_sessions[np.random.choice(range(training_num), batch_size)], learning_rate,
                                  True)
        # agent.learn_from_sessions(train_sessions, True)
        agent.validate_from_sessions(finished_sessions[training_num:])
        # print("Saving model...")
        # agent.save_model(save_dir)


def human_vs_human():
    board_size = (15, 15)
    player1 = human_agent.HumanAgent("human1", board_size, 1)
    player2 = human_agent.HumanAgent("human2", board_size, -1)
    model_path = os.path.join("model")
    run_with_agents(10, player1, player2, model_path, model_path, True, True)


def import_games():
    years = range(2009, 2017, 1)
    for y in years:
        game_folder = '/Users/wei.cui/Downloads/gomocup%dresults/standard' % y
        print('importing games from ', game_folder)
        training.psq_game.convert_psq_files(game_folder, 'data/games.bdt')
        for i in range(1, 4, 1):
            game_folder = '/Users/wei.cui/Downloads/gomocup%dresults/freestyle%d' % (y, i)
            print('importing games from ', game_folder)
            training.psq_game.convert_psq_files(game_folder, 'data/games.bdt')


# set_logging(logging.DEBUG)
# human_vs_human()
# import_games()
# training.bdt_game.replay_games('data/games.bdt')
# training.psq_game.convert_psq_files('../history/standard', 'data/gomocup-2016.bdt')
# training.psq_game.convert_psq_files('/Users/wei.cui/Downloads/gomocup2009results/Standard', 'data/games.bdt')
# train_with_games("/output/data", "/output/model")
# train_with_raw_data("data", "/output/model")
# train_with_raw_data("data", "../history/gomocup-2016-5/model")
# train_with_games("../history/gomocup-2016/data", "../history/gomocup-2016/model")
# ai_vs_human("../history/gomocup-2016-7", True, True, False)

# train_with_games("../history/value-policy-ai-1", 'data/games.bdt', True, True)
# train_with_games("../history/value-policy-ai-4", 'data/games.bdt', True, True)
# train_with_games("value-net-agent", 'data/games.bdt', True, True)
# train_with_games("/output", 'data/games.bdt', True)
# ai2_vs_human("../history/gcloud-models/value-net-agent", True, False, False)
# ai_vs_human("../history/gomocup-2016-6", True, False, False)
# ai_vs_ai("../history/ai_vs_ai", True)

# train_evaluator(1000, 100)
#run_actor_critic_agent(10)

alpha_training.run(100, 10, '../history/alpha_zero/model')

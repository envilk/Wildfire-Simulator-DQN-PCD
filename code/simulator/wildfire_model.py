# python libraries
import copy
import os
import sys
import time

import mesa
import statistics
import matplotlib.pyplot as plt
import matplotlib as mpl

# own python modules

import agents

sys.path.append('../')
sys.path.append('../DQN/')

import nn_definitions

from common_fixed_variables import *
from paths import *


class WildFireModel(mesa.Model):

    def __init__(self):

        plt.ion()

        # wildfiremodel exclusive vars

        self.new_direction_counter = None
        self.datacollector = None
        self.grid = None
        self.unique_agents_id = None
        self.eval_step_counter = 0
        self.eval_ac_reward = 0
        self.EPISODE_REWARD_MEANS = []

        # in widlfiremodel, but used by DQN

        self.END_EVAL = False
        self.new_direction = None
        self.NUM_AGENTS = NUM_AGENTS  # Number of UAV (IMPORTANT, for auto-eval max number required)
        print(self.NUM_AGENTS)
        self.EPISODE_REWARD = []
        self.times = []
        self.interactions_counter = 0

        # FIXME extracted from 'nn_learning_process.py', might not be correct
        #  | used when inference is activated for both, interface and auto evaluation
        self.policy_net = nn_definitions.DQN().to(DEVICE)
        if INFERENCE and not RNN and not HORIZON_EVAL:
            self.policy_net.load_state_dict(torch.load('../../results/DQN/training_checkpoints/' + str(self.NUM_AGENTS)
                                                       + '_drones_checkpoint_policy_net15000.pth'))
            self.policy_net.eval()
            print('model correctly loaded')

        self.reset()

    def reset(self):
        self.unique_agents_id = 0
        # Inverted width and height order, because of matrix accessing purposes, like in many examples:
        #   https://snyk.io/advisor/python/Mesa/functions/mesa.space.MultiGrid
        self.grid = mesa.space.MultiGrid(HEIGHT, WIDTH, False)
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.set_fire_agents()
        self.wind = agents.Wind()
        # set two UAV
        x_center = int(HEIGHT / 2)
        y_center = int(WIDTH / 2)

        self.new_direction_counter = 0
        self.eval_step_counter = 0
        self.eval_ac_reward = 0

        for a in range(0, self.NUM_AGENTS):
            aux_UAV = agents.UAV(self.unique_agents_id, self)
            y_center += a if a % 2 == 0 else -a
            self.grid.place_agent(aux_UAV, (x_center, y_center + 1))
            self.schedule.add(aux_UAV)
            self.unique_agents_id += 1

        self.datacollector = mesa.DataCollector()
        self.new_direction = [0 for a in range(0, self.NUM_AGENTS)]
        self.times = []

    def plot_durations(self, show_result=False):
        plt.figure(1)
        if show_result:
            plt.title('Result')
            with open(ROOT_RESULTS_DQN_TRAINING_RESULTS + str(self.NUM_AGENTS) + 'UAV_training_results.txt', 'w') as f:
                print(self.EPISODE_REWARD_MEANS)
                to_write = [str(reward) + '\n' for reward in self.EPISODE_REWARD_MEANS]
                f.writelines(to_write)
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        c = numpy.arange(0, self.NUM_AGENTS + 2)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap_rewards = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        cmap_rewards_means = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
        cmap_rewards.set_array([])
        cmap_rewards_means.set_array([])

        if INFERENCE:
            for agent_idx in range(0, self.NUM_AGENTS):
                to_plot = [reward[agent_idx] for reward in self.EPISODE_REWARD]
                plt.plot(to_plot, c=cmap_rewards.to_rgba(agent_idx + 1))
        else:
            min_num = 100
            # for each agent, extract mean of last 'min_num' elements of the 'self.EPISODE_REWARD' list. Then, append
            # to 'self.EPISODE_REWARD',
            aux_means = [0 for agent_idx in range(0, self.NUM_AGENTS)]
            if len(self.EPISODE_REWARD) >= min_num + 1:  # extract means
                aux_means = [statistics.mean([reward[agent_idx] for reward in self.EPISODE_REWARD[-min_num:]])
                             for agent_idx in range(0, self.NUM_AGENTS)]
            self.EPISODE_REWARD_MEANS.append(aux_means)  # store mean
            for agent_idx in range(0, self.NUM_AGENTS):  # plot it
                to_plot = [means[agent_idx] for means in self.EPISODE_REWARD_MEANS]
                plt.plot(to_plot, c=cmap_rewards_means.to_rgba(agent_idx + 1))
        plt.pause(0.001)  # pause a bit so that plots are updated

    def set_fire_agents(self):
        x_c = int(HEIGHT / 2)
        y_c = int(WIDTH / 2)
        x = [x_c]  # , x_c + 1, x_c - 1
        y = [y_c]  # , y_c + 1, y_c - 1
        for i in range(HEIGHT):
            for j in range(WIDTH):
                # decides to put a "tree" (fire agent) or not
                if SYSTEM_RANDOM.random() < DENSITY_PROB or (i in x and j in y):  # if prob or in center
                    if i in x and j in y:
                        self.new_fire_agent(i, j, True)
                    else:
                        self.new_fire_agent(i, j, False)

    def new_fire_agent(self, pos_x, pos_y, burning):
        source_fire = agents.Fire(self.unique_agents_id, self, burning)
        self.unique_agents_id += 1
        self.schedule.add(source_fire)
        self.grid.place_agent(source_fire, tuple([pos_x, pos_y]))

    def normalize(self, to_normalize, upper, multiplier, subtractor):
        return ((to_normalize / upper) * multiplier) - subtractor

    def set_drone_dirs(self):
        self.new_direction_counter = 0
        for agent in self.schedule.agents:
            if type(agent) is agents.UAV:
                agent.selected_dir = self.new_direction[self.new_direction_counter]
                self.new_direction_counter += 1

    def state(self):
        states = []
        rewards = []
        states_positions = []
        UAV_positions = []
        rewards_for_evaluation = []
        for agent in self.schedule.agents:
            if type(agent) is agents.UAV:
                surrounding_states, reward, positions = agent.surrounding_states()
                states.append(surrounding_states)
                aux_reward = self.normalize(float(reward), N_OBSERVATIONS, 1, 0)
                rewards.append(aux_reward)
                rewards_for_evaluation.append(copy.deepcopy(aux_reward))
                states_positions.append(positions)
                UAV_positions.append(agent.pos)

        # if there are three agents: intersection [(1-2), (1-3)], [(2-1), (2-3)], [(3-1), (3-2)]
        # Basically the sentence shown before means that an UAV receives a -1 reward when it
        # overlaps one cell in its observation area with another UAV
        # FINNISH COMMENT
        for i in range(0, len(states_positions)):
            reward_discount_counter = 0
            aux_states_positions = states_positions.copy()
            aux_final_states = []
            del aux_states_positions[i]

            for st in aux_states_positions:
                aux_final_states.extend(set(states_positions[i]) & set(st))

            reward_discount_counter += (len(set(aux_final_states)) / N_OBSERVATIONS)
            rewards[i] -= reward_discount_counter

        # when UAV reaches edge or corner, takes less surrounding states, so fulfilling the vector till
        # maximum amount of observation is necessary. It is IMPORTANT to mention that this COULD AFFECT to
        # the correct behaviour of drones when these are trying to maintain distance with other, because of
        # less area is taking into account. Either way, in most cases this is expendable.
        for st, _ in enumerate(states):
            counter = len(states[st])
            for i in range(counter, N_OBSERVATIONS):
                states[st].append(0)
        return states, rewards, rewards_for_evaluation

    def select_action_inference(self, state):
        with torch.no_grad():
            # t.argmax() will return the index of the max action prob (int index).
            # from the four defined (north, east, south, west)
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            dirs = self.policy_net(state)
            returned_list = []

            # needed to be developed like this, because policy.net returns different depending on the dimensions
            if self.NUM_AGENTS < 2:
                _, idx = dirs.topk(1)
                # 50% takes first best direction for that cell, other 50% takes second best
                returned = idx[0]  # if sample > 0.3 else idx[1]
                returned_list.append(returned.view(1, 1))
                # print(returned, idx)
            else:
                for dir in dirs:
                    _, idx = dir.topk(1)
                    # 50% takes first best direction for that cell, other 50% takes second best
                    returned = idx[0]  # if sample > 0.3 else idx[1]
                    returned_list.append(returned.view(1, 1))

            return torch.tensor(returned_list)

    def calculate_evaluation_rewards(self, reward, rewards_for_evaluation):
        if INTERFACE == 'Automatic':  # for setting metric explained in paper SEAMS2024
            aux_reward = rewards_for_evaluation
        else:
            aux_reward = reward
        self.EPISODE_REWARD.append(aux_reward)
        aux_reward = torch.tensor([aux_reward], device=DEVICE)
        self.eval_ac_reward += aux_reward

    def count_security_distance_overpassing(self):
        count = 0
        UAV_agents = []

        for agent in self.schedule.agents:
            if type(agent) is agents.UAV:
                UAV_agents.append(agent)

        for idx, agent in enumerate(UAV_agents):
            aux_agents_positions = UAV_agents.copy()
            del aux_agents_positions[idx]

            for a in aux_agents_positions:
                x1 = agent.pos[0]
                y1 = agent.pos[1]
                x2 = a.pos[0]
                y2 = a.pos[1]
                distance = euclidean_distance(x1, y1, x2, y2) # 0 agent just to take euclidean function, no reason for any UAV
                if distance < SECURITY_DISTANCE:
                    count += 1

        return count

    def calculate_metrics(self, UAV_idx, auto, label, num_run, strings_to_write_eval, overall_interactions, type):
        if not auto:
            self.plot_durations(show_result=True)
        if self.eval_step_counter == BATCH_SIZE:
            self.END_EVAL = True
            if auto:
                root = ROOT_RESULTS_DQN_INFERENCE_RESULTS + str(UAV_idx + 1) + 'UAV_run_rewards' + '/'
                if not os.path.isdir(root):
                    os.mkdir(root)
                title = root + str(num_run) + '_EVAL_drones_pyfoo.svg'
                self.plot_durations()
                strings_to_write_eval.append(str(*self.eval_ac_reward.tolist()) + '\n')

                overall_interactions.append(
                    str(self.interactions_counter / 2) + '\n')  # divided by two, to remove duplicates interactions
            else:
                title = ROOT_RESULTS_DQN_INFERENCE_RESULTS + str(self.NUM_AGENTS) + '_EVAL_drones_pyfoo.svg'
                print('EVAL AC REWARD: ', *self.eval_ac_reward.tolist())
            plt.savefig(title)

    def step(self, UAV_idx=-1, label=-1, type='', num_run=0, auto=False, strings_to_write_eval='',
             overall_interactions=''):
        self.datacollector.collect(self)
        if UAV_idx == 1:
            pass
        if sum(isinstance(i, agents.UAV) for i in self.schedule.agents) > 0:
            if INFERENCE:
                state, _, _ = self.state()  # s_t
                if RNN:
                    self.new_direction = [SYSTEM_RANDOM.choice(range(0, N_ACTIONS))
                                          for i in range(0, self.NUM_AGENTS)]  # a_t
                else:
                    start = time.time()
                    self.new_direction = self.select_action_inference(state)  # a_t
                    end = time.time() - start
                    self.times.append(end)

                _, reward, rewards_for_evaluation = self.state()  # r_t+1
                # print(reward, self.new_direction)

                self.calculate_evaluation_rewards(reward, rewards_for_evaluation)

                if INTERFACE == 'Automatic':
                    self.interactions_counter += self.count_security_distance_overpassing()

                self.calculate_metrics(UAV_idx, auto, label, num_run, strings_to_write_eval,
                                       overall_interactions, type)

                self.eval_step_counter += 1
            self.set_drone_dirs()
        self.schedule.step()  # self.new_direction is used to execute previous obtained a_t

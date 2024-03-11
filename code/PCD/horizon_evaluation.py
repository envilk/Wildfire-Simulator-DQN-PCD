import itertools
import os
import sys
import time
import mesa
import numpy
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import h5py

sys.path.append('../')
sys.path.append('../simulator/')
sys.path.append('../mesa_addons/')

import agents, wildfire_model
from Space_Grid_Setter_Getter import MultiGrid
from common_fixed_variables import *
from paths import *


class HorizonEvaluation(wildfire_model.WildFireModel):

    # set the enviroment (grid, schedule, fire agents), new UAVs, eval parameters (step counter, ac reward,
    # eval times), data collector, 'new_directions' variable for UAVs
    def __init__(self):
        self.grids = None
        self.new_direction = None
        self.datacollector = None
        self.agents_positions = None
        self.agents = None
        self.eval_ac_reward = None
        self.wind = None
        self.grid = None
        self.unique_agents_id = None
        self.NUM_AGENTS = NUM_AGENTS
        self.times = []

        self.reset()

    def reset(self):
        self.unique_agents_id = 0
        # Inverted width and height order, because of matrix accessing purposes, like in many examples:
        #   https://snyk.io/advisor/python/Mesa/functions/mesa.space.MultiGrid
        self.grid = MultiGrid(HEIGHT, WIDTH, False)
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.set_fire_agents()
        self.wind = agents.Wind()
        # set two UAV
        x_center = int(HEIGHT / 2)
        y_center = int(WIDTH / 2)

        self.eval_ac_reward = 0
        self.agents = []
        self.agents_positions = []
        self.times = []

        for a in range(0, self.NUM_AGENTS):
            aux_UAV = agents.UAV(self.unique_agents_id, self)
            y_center += a if a % 2 == 0 else -a
            self.agents.append(aux_UAV)  # FIXME: ñapa?, arreglar porque solo se usa en evaluation()?
            self.agents_positions.append((x_center, y_center + 1))
            self.unique_agents_id += 1

        self.datacollector = mesa.DataCollector()
        self.new_direction = [0 for a in range(0, self.NUM_AGENTS)]

    def plot_durations(self, num_max_uavs):
        plt.figure(1)
        plt.clf()
        plt.title('Result')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        c = numpy.arange(0, self.NUM_AGENTS + 2)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap_rewards = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        cmap_rewards_means = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
        cmap_rewards.set_array([])
        cmap_rewards_means.set_array([])

        for agent_idx in range(0, num_max_uavs):
            to_plot = [reward[agent_idx] for reward in self.EPISODE_REWARD]
            plt.plot(to_plot, c=cmap_rewards.to_rgba(agent_idx + 1))
        plt.pause(0.001)  # pause a bit so that plots are updated

    def count_security_distance_overpassing(self, idx, num_max_uavs):
        count = 0

        aux_agents_positions = self.agents_positions[:num_max_uavs].copy()
        del aux_agents_positions[idx]

        for a_pos in aux_agents_positions:
            x1 = self.agents_positions[idx][0]
            y1 = self.agents_positions[idx][1]
            x2 = a_pos[0]
            y2 = a_pos[1]
            distance = euclidean_distance(x1, y1, x2,
                                          y2)  # 0 agent just to take euclidean function, no reason for any UAV
            if distance < SECURITY_DISTANCE:
                count += 1

        return count

    def evaluation(self):
        max_agents = self.NUM_AGENTS

        times = []  # measure runs spent execution time, and done FOR EXECUTING ONLY WITH "num_runs = 1"
        print(times)

        for UAV_idx in range(0,
                             max_agents):  # "max_agents" must go to the max number of agents on training (self.NUM_AGENTS)
            num_max_uavs = UAV_idx + 1
            num_runs = 1
            self.strings_to_write_eval = []
            self.overall_interactions = []
            batches = []

            for i in range(0, num_runs):
                interactions_counter = 0
                print('-------- NUM_RUN: ', i, '--------')
                self.EPISODE_REWARD = []
                self.reset()
                self.obtain_grids(num_run=i)

                simulation = []
                for t in itertools.count():
                    print('-------- STEP: ', t, '--------')
                    # if done:
                    if t == BATCH_SIZE:
                        break

                    """
                    interactions_counter, states_and_position_mixed = self.evaluation_step(UAV_idx, i,
                                                                                           interactions_counter,
                                                                                           num_max_uavs, t)
                    states_and_position_mixed_numpy = numpy.array(states_and_position_mixed)
                    simulation.append(states_and_position_mixed_numpy)
                    print(states_and_position_mixed_numpy.shape)
                    """
                    interactions_counter, rewards_and_actions_mixed = self.evaluation_step(UAV_idx, i,
                                                                                           interactions_counter,
                                                                                           num_max_uavs, t)
                    rewards_and_actions_mixed_numpy = numpy.array(rewards_and_actions_mixed)
                    simulation.append(rewards_and_actions_mixed_numpy)
                    print(rewards_and_actions_mixed_numpy.shape)

                simulation_numpy = numpy.stack(simulation, axis=0)
                batches.append(simulation_numpy)
                print(simulation_numpy.shape)

                times.append(self.times)

                self.calculate_metrics(i, interactions_counter, num_max_uavs)

            batches_numpy = numpy.stack(batches, axis=0)
            print(batches_numpy.shape)

            self.write_h5py_file_LSTM_dataset(batches_numpy, num_max_uavs)
            self.write_metrics_files(num_max_uavs)

        print("--- TIMES ---")
        [print(times[idx]) for idx in range(0, max_agents)]

    def write_h5py_file_LSTM_dataset(self, data, num_max_uavs):
        print()
        print()
        print('PRINTING DATASET')
        print()
        print()
        print(data.shape, data)
        # Create a new HDF5 file
        with h5py.File('dataset_' + str(num_max_uavs) + 'UAV.hdf5', 'w') as f:
            # Create a dataset within the file to store your data
            dset = f.create_dataset('data', data=data)

    def write_metrics_files(self, num_max_uavs):
        # CHANGE STR(0) IF PARALLELIZATION IS NEEDED. PUT AS MANY NUMBERS AS CPUS ARE USED
        # (THIS AVOID COPYING PROJECTS UNNECESSARILY).
        with open(ROOT_RESULTS_PCD_INFERENCE_RESULTS + str(
                num_max_uavs) + 'UAV.txt',
                  'w') as f:
            f.writelines(self.strings_to_write_eval)
        with open(ROOT_RESULTS_PCD_INFERENCE_RESULTS + 'COUNTER_' + str(
                num_max_uavs) + 'UAV.txt',
                  'w') as f2:
            f2.writelines(self.overall_interactions)

    def calculate_metrics(self, i, interactions_counter, num_max_uavs):
        # plot individual rewards at the end of each run (complete simulation)
        plt.figure(1)
        root = ROOT_RESULTS_PCD_INFERENCE_RESULTS + str(
            num_max_uavs) + 'UAV_run_rewards' + '/'
        if not os.path.isdir(root):
            os.mkdir(root)
        title = root + str(i) + '_EVAL_drones_pyfoo.svg'
        self.plot_durations(num_max_uavs)
        self.strings_to_write_eval.append(str(self.eval_ac_reward.tolist()) + '\n')
        plt.savefig(title)
        print('FINAL-', interactions_counter)
        self.overall_interactions.append(
            str(interactions_counter / 2) + '\n')  # divided by two, to remove duplicates interactions

    def evaluation_step(self, UAV_idx, i, interactions_counter, num_max_uavs, t):
        # 1: visualize fire state
        self.reset_step_horizon_conf(UAV_idx, t, False)
        self.visualize_grid_evaluation(UAV_idx=UAV_idx, num_run=i, instant=t)
        self.grid.remove_UAVs()

        start = time.time()
        # 2: calculate accumulated SUM rewards, within the horizon step "h"
        best_trajectory = self.best_trajectory_within_horizon(num_max_uavs, UAV_idx, i, t)
        end = time.time() - start
        self.times.append(end)
        reward_to_append = []

        for idx, _ in enumerate(self.agents[:num_max_uavs]):
            interactions_counter += self.count_security_distance_overpassing(idx, num_max_uavs)
            reward_to_append.append(best_trajectory[idx]["rewards_for_evaluation"][0])
        self.EPISODE_REWARD.append(reward_to_append)
        self.eval_ac_reward += torch.tensor(reward_to_append, device=DEVICE)

        # 4: reset agents positions for the next step "t" with best dirs
        for idx, a in enumerate(self.agents[:num_max_uavs]):
            self.agents_new_pos(idx, best_trajectory[idx]["actions"][0])

        # 5: build dataset
        """
        states_and_position_mixed = self.build_LSTM_dataset_instance(best_trajectory, num_max_uavs)

        return interactions_counter, states_and_position_mixed
        """
        rewards_and_actions_mixed = self.build_LSTM_dataset_instance(best_trajectory, num_max_uavs)

        return interactions_counter, rewards_and_actions_mixed

    def build_LSTM_dataset_instance(self, best_trajectory, num_max_uavs):
        """
        list_1 = [dict["states"][0] for dict in best_trajectory]
        list_2 = []

        for pos in self.agents_positions[:num_max_uavs]:
            print(pos[0], pos[1])
            pos_aux_1 = str(pos[0])
            pos_aux_2 = str(pos[1])

            list_2.append([int(pos_aux_1), int(pos_aux_2)])

        states_and_position_mixed = []
        for state, pos in zip(list_1, list_2):
            states_and_position_mixed.append(state + pos)
        print(states_and_position_mixed)

        return states_and_position_mixed
        """
        list_1 = [dict["rewards"][0] for dict in best_trajectory]
        list_2 = [dict["actions"][0] for dict in best_trajectory]

        rewards_and_actions_mixed = []
        for reward, action in zip(list_1, list_2):
            rewards_and_actions_mixed.append([reward, action])
        print(rewards_and_actions_mixed)

        return rewards_and_actions_mixed


    def best_trajectory_within_horizon(self, num_max_uavs, UAV_idx, i, t):
        best_mean_reward = float('-inf')
        best_trajectory = None
        horizon = 1
        Z = 15
        for z in range(Z):
            self.reset_step_horizon_conf(UAV_idx, t, False)
            best_coordinate_descent = [{"actions": [], "rewards": [], "rewards_for_evaluation": [], "states": []}
                                       for _ in range(num_max_uavs)]
            for step in range(horizon):
                # TODO self.visualize_grid_evaluation(UAV_idx=UAV_idx, num_run=i, instant=t)
                for idx, a in enumerate(self.agents[:num_max_uavs]):
                    moved_once_in_loop = self.select_and_move_agent(a, best_coordinate_descent, idx, num_max_uavs, step)

                    self.finalize_move(UAV_idx, a, best_coordinate_descent, idx, moved_once_in_loop, num_max_uavs, step,
                                       t)

            # 2.5: remove UAVs from grids (placed again at the beginning of the combinations loop)
            for step in range(horizon + 1):
                if t + step < BATCH_SIZE:
                    self.reset_step_horizon_conf(UAV_idx, t + step, False)
            # calculate accumulated reward for each trajectory, to compare them
            ac_rewards = [sum(best_coordinate_descent[idx]["rewards"]) for idx in range(num_max_uavs)]
            mean_reward = sum(ac_rewards) / len(ac_rewards)
            # apply noise based on a parametrized normal distribution
            mean_reward = self.apply_noise(mean_reward)

            # pick the best trajectory based on the rest
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_trajectory = best_coordinate_descent

        return best_trajectory

    def finalize_move(self, UAV_idx, a, best_coordinate_descent, idx, moved_once_in_loop, num_max_uavs, step, t):
        # reset for each tried action
        aux_horizon = t + step
        if aux_horizon + 1 < BATCH_SIZE:
            # move if possible to the best chosen action from the possible ones, then do the same with the rest of the agents. If UAV was moved towards any direction, then moved, if it wasn't moved, there is not best action, so just don't move.
            if moved_once_in_loop:
                a.selected_dir = best_coordinate_descent[idx]["actions"][step]
                a.move()
            # if we are checking last UAV, then change grid for next step in horizon, otherwise, maintain same grid
            if idx == num_max_uavs - 1:
                aux_horizon += 1
            self.reset_step_horizon_conf(UAV_idx, aux_horizon, True)

            # TODO self.visualize_grid_evaluation(UAV_idx=UAV_idx, num_run=i, instant=t)

    def select_and_move_agent(self, a, best_coordinate_descent, idx, num_max_uavs, step):
        oposite_idx = 2  # to move agent the opposite way
        tested_first_possible_action = False
        moved_once_in_loop = False
        # has to be reset each time, otherwise in the step 't + 1' is comparing with the last best reward, which can lead into not finding any optimal action in the time step 't' (which doesn't make any sense)
        best_reward = [float('-inf') for _ in self.agents[:num_max_uavs]]
        possible_actions = [a for a in range(N_ACTIONS)]
        if step == 0: possible_actions = [SYSTEM_RANDOM.choice(possible_actions)]
        for action in possible_actions:
            a.selected_dir = action
            moved = a.move()
            # TODO self.visualize_grid_evaluation(UAV_idx=UAV_idx, num_run=i, instant=t)

            states, reward, rewards_for_evaluation = self.state()

            if step == 0:  # random start in coordinate descent iterations
                self.append_agent_step(states, action, best_coordinate_descent, idx, reward,
                                       rewards_for_evaluation)
            else:  # try all actions, like normally with all 'Z' iterations
                tested_first_possible_action = self.update_best_reward(states, action, best_coordinate_descent,
                                                                       best_reward,
                                                                       idx, reward,
                                                                       rewards_for_evaluation, step,
                                                                       tested_first_possible_action)
            # this if statement helps to indicate whether the UAV must be moved or not to the opposite direction, depending on if before it could be moved. If not, it shouldn't be moved to any other position (done when testing all actions as well, in the else statement below)
            moved_once_in_loop = self.check_if_agent_moved(a, action, moved, moved_once_in_loop,
                                                           oposite_idx, possible_actions)
        return moved_once_in_loop

    def check_if_agent_moved(self, a, action, moved, moved_once_in_loop, oposite_idx, possible_actions):
        if moved:
            moved_once_in_loop = True
            # move opposite direction (original state) to try new possible actions
            a.selected_dir = (action + oposite_idx) % len(possible_actions)
            a.move()
            # TODO self.visualize_grid_evaluation(UAV_idx=UAV_idx, num_run=i, instant=t)
        return moved_once_in_loop

    def update_best_reward(self, states, action, best_coordinate_descent, best_reward, idx, reward,
                           rewards_for_evaluation,
                           step, tested_first_possible_action):
        if reward[idx] > best_reward[idx]:
            best_reward[idx] = reward[idx]
            if not tested_first_possible_action:
                tested_first_possible_action = True
                self.append_agent_step(states, action, best_coordinate_descent, idx, reward,
                                       rewards_for_evaluation)
            else:
                best_coordinate_descent[idx]["rewards_for_evaluation"][step] = rewards_for_evaluation[idx]
                best_coordinate_descent[idx]["rewards"][step] = reward[idx]
                best_coordinate_descent[idx]["actions"][step] = action
                best_coordinate_descent[idx]["states"][step] = states[idx]
        return tested_first_possible_action

    def append_agent_step(self, states, action, best_coordinate_descent, idx, reward, rewards_for_evaluation):
        best_coordinate_descent[idx]["rewards_for_evaluation"].append(rewards_for_evaluation[idx])
        best_coordinate_descent[idx]["rewards"].append(reward[idx])
        best_coordinate_descent[idx]["actions"].append(action)
        best_coordinate_descent[idx]["states"].append(states[idx])

    def apply_noise(self, reward):
        if NOISE:
            noise = numpy.random.normal(reward, OMEGA)
            reward += noise
        return reward

    def agents_new_pos(self, UAV_idx, new_dir):
        # directions = [0, 1, 2, 3]  # right, down, left, up
        # self.selected_dir = random.choice(directions)
        move_x = [1, 0, -1, 0]
        move_y = [0, -1, 0, 1]

        # print(self.agents_positions[UAV_idx], move_x[new_dir])
        pos_to_move = (
            self.agents_positions[UAV_idx][0] + move_x[new_dir], self.agents_positions[UAV_idx][1] + move_y[new_dir])
        if not self.grid.out_of_bounds(pos_to_move) and (pos_to_move not in self.agents_positions):
            self.agents_positions[UAV_idx] = pos_to_move

    def visualize_grid_evaluation(self, UAV_idx=-1, num_run=-1, instant=-1):
        agent_counts = numpy.zeros((self.grid.width, self.grid.height))
        for cell_content, (x, y) in self.grid.coord_iter():
            agent_count = len(cell_content)
            # draw UAV
            if agent_count == 2:
                agent_count = 3
            # cell with smoke active
            if agent_count == 1 and cell_content[0].smoke.is_smoke_active():
                agent_count = 2
            # cell is burning
            if agent_count == 1 and (not cell_content[0].is_burning() or cell_content[0].get_fuel() == 0):
                agent_count = 0
            agent_counts[y][x] = agent_count
        # Plot using seaborn, with a size of 5x5
        plt.figure(2)
        plt.clf()
        g = sns.heatmap(agent_counts, cmap="viridis", annot=False, cbar=False, square=True)
        # invert y-axis to take same bottom-left reference as with mesa grid
        g.invert_yaxis()
        g.figure.set_size_inches(15, 15)
        g.set(title="Number of agents on each cell of the grid")

        # plt.show(block=True)

        path = ROOT_RESULTS_PCD_INFERENCE_GRID_INSTANTS + str(UAV_idx + 1) + 'UAV/' + str(num_run) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        title = path + 't_' + str(instant) + '.png'
        plt.savefig(title)

    def reset_step_horizon_conf(self, UAV_idx, t, iterating):
        # 0: set actual grid, and reset scheduler
        if self.grid:
            del self.grid
        self.grid = self.grids[t]['grid']
        # 1: copy agents positions for t step with agents_position
        UAV = False
        aux_agents = []
        for cell, (x, y) in self.grid.coord_iter():
            if len(cell) > 1:
                for c in cell:
                    if type(c) is agents.UAV:
                        UAV = True
                        aux_agents.append(c)

        for idx, a in enumerate(self.agents[:UAV_idx + 1]):
            if not UAV:
                if iterating:
                    # print('not uav, iter')
                    self.grid.place_agent(a, a.pos)
                else:
                    # print('not uav, not iter')
                    self.grid.place_agent(a, self.agents_positions[idx])
            elif not iterating:
                self.grid.remove_UAVs()

        self.reset_scheduler_for_step_horizon()

    def reset_scheduler_for_step_horizon(self):
        self.schedule = mesa.time.SimultaneousActivation(self)
        ids = []
        for coord in self.grid.coord_iter():
            if len(coord[0]) > 1:
                for e in coord[0]:
                    if e.unique_id not in ids:
                        self.schedule.add(e)
                        ids.append(e.unique_id)
            elif coord[0][0].unique_id not in ids:
                self.schedule.add(coord[0][0])
                ids.append(coord[0][0].unique_id)

    def obtain_grids(self, _type='', num_run=0):
        if self.grids:  # make sure copies don't explode memory
            del self.grids
        self.grids = []

        for t in itertools.count():
            self.reset_scheduler_for_step_horizon()
            aux_grid = self.grid

            agents_state_list_to_show = []
            for coord in aux_grid.coord_iter():
                agents_state_list_to_show.append(int(coord[0][0].is_burning()))
            aux_grid_info = {"step": t, "overall_state": sum(agents_state_list_to_show), "grid": aux_grid}

            self.grids.append(aux_grid_info)
            self.step()

            # Resetting self.grid on each iteration, in order to use the rest of the methods easily, not creating new ones
            _copy = self.grid.get_copy()
            del self.grid
            self.grid = MultiGrid(HEIGHT, WIDTH, False)
            self.grid.set_grid(_copy)

            # if done:
            if BATCH_SIZE == t:
                break

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

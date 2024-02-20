# python libraries

import torch.nn as nn
import torch.optim as optim
import collections
import numpy
import itertools
import math
import time
import os
import sys
import matplotlib.pyplot as plt

# own python modules

import nn_definitions
import data_structures

sys.path.append('../')
sys.path.append('../simulator/')

from common_fixed_variables import *
from paths import *


class DQNLearning:

    def __init__(self, wildfiremodel=None):

        self.model = wildfiremodel
        self.transition = collections.namedtuple('Transition',
                                                 ('state', 'action', 'next_state', 'reward'))
        self.i_episode = 0
        self.steps_done = 0
        self.checkpoint = 250
        self.time_elapsed_checkpoint = 1000
        self.num_episodes = 15001 if torch.cuda.is_available() else 50
        self.policy_net = nn_definitions.DQN().to(DEVICE)
        self.target_net = nn_definitions.DQN().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = data_structures.ReplayMemory(10000, self.transition)

    def select_action_training(self, state):
        sample = SYSTEM_RANDOM.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:  # epsilon-greedy policy
            with torch.no_grad():
                # t.argmax() will return the index of the max action prob (int index).
                # from the four defined (north, east, south, west)
                action_probabilities = self.policy_net(state).tolist()  # .argmax().view(1, 1)
                if self.model.NUM_AGENTS >= 2:
                    actions = [numpy.argmax(a_p) for a_p in action_probabilities]
                else:
                    actions = [numpy.argmax(action_probabilities)]
                return torch.tensor(actions, device=DEVICE, dtype=torch.long)
        else:
            return torch.tensor(
                [SYSTEM_RANDOM.choice(range(0, N_ACTIONS)) for i in range(0, self.model.NUM_AGENTS)],
                device=DEVICE, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample()

        dimension = 2 if self.model.NUM_AGENTS >= 2 else 1
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        # x -> (BATCH_SIZE, NUM_AGENTS, NUM_OBSERVATIONS) | torch.Size([90, 3, 289])
        state_batch = torch.stack(list(batch.state), dim=0)
        # x -> (BATCH_SIZE, NUM_AGENTS) | torch.Size([90, 3])
        action_batch = torch.stack(list(batch.action), dim=0)
        # x -> (BATCH_SIZE, 1, NUM_AGENTS) | torch.Size([90, 1, 3])
        reward_batch = torch.stack(list(batch.reward), dim=0)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        action_batch = action_batch.unsqueeze(dimension) if self.model.NUM_AGENTS >= 2 else action_batch
        sav = self.policy_net(state_batch)
        state_action_values = sav.gather(dimension, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = self.target_net(non_final_next_states).max(dimension)[0]
        # Compute the expected Q values
        if self.model.NUM_AGENTS >= 2:
            expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch
        else:
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

        # Compute Huber loss
        criterion = nn.SmoothL1Loss(reduction='sum')
        loss = criterion(state_action_values.squeeze(), expected_state_action_values.squeeze())

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        start = time.perf_counter()

        for episode in range(self.num_episodes):
            self.i_episode = episode
            # Initialize the environment and get it's state
            self.model.reset()
            state, _, _ = self.model.state()
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            episode_rewards = 0
            for t in itertools.count():
                # Input -> state [self.NUM_OBSERVATIONS] | Output -> action [self.NUM_AGENTS] p.e. I[25] -> O[2]
                actions = self.select_action_training(state)

                # Output -> state [self.NUM_OBSERVATIONS], reward [self.NUM_AGENTS] p.e. I[25] -> O[2]
                state, _, _ = self.model.state()
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
                self.model.new_direction = actions
                self.model.step(-1, -1, -1)
                next_state, reward, _ = self.model.state()

                reward = torch.tensor([reward], device=DEVICE)
                episode_rewards += reward
                next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)

                # Store the transition in memory Transition([self.NUM_AGENTS, self.NUM_OBSERVATIONS],
                # [1, self.NUM_AGENTS], [self.NUM_AGENTS, self.NUM_OBSERVATIONS], [self.NUM_AGENTS])
                # p.e. T([2, 25], [1, 2], [2, 25], [2])
                self.memory.push(state, actions, next_state, reward)

                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                            1 - TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                # if done:
                if t == BATCH_SIZE:
                    # print(episode_rewards)
                    self.model.EPISODE_REWARD.append(*episode_rewards.tolist())
                    self.model.plot_durations()
                    break

            # removes the need of waiting until the training process is done
            if self.i_episode % self.checkpoint == 0:
                root = ROOT_RESULTS_DQN_TRAINING_CHECKPOINTS
                if os.path.exists(root):
                    torch.save(self.policy_net.state_dict(),
                               root + str(self.model.NUM_AGENTS) + '_drones_checkpoint_policy_net'
                               + str(self.i_episode) + '.pth')
                else:
                    os.mkdir(root)

            # allows to print time elapsed each certain time steps
            if self.i_episode % self.time_elapsed_checkpoint == 0:
                plt.savefig(ROOT_RESULTS_DQN_TRAINING_RESULTS + str(self.model.NUM_AGENTS) + '_' + str(self.i_episode) + '_drones_pyfoo.svg')
                end = time.perf_counter() - start
                print('EPISODE: ', self.i_episode, 'Elapsed time:', end)

        # torch.save(self.policy_net.state_dict(), 'policy_net.pth')
        print('Complete')

        self.model.plot_durations(show_result=True)
        plt.ioff()
        plt.savefig(str(self.model.NUM_AGENTS) + '_drones_pyfoo.svg')
        # plt.show()

    def evaluation(self, type='entrenamiento'):
        max_agents = self.model.NUM_AGENTS

        times = []  # measure runs spent execution time, and done FOR EXECUTING ONLY WITH "num_runs = 1"
        print(times)

        for UAV_idx in range(0,
                             max_agents):  # "max_agents" must go to the max number of agents on training (self.NUM_AGENTS)
            self.model.NUM_AGENTS = UAV_idx + 1
            if type == 'entrenamiento':  # to not waste time in loading models when predictor activated
                to_load = ROOT_RESULTS_DQN_TRAINING_CHECKPOINTS + str(
                    UAV_idx + 1) + '_drones_checkpoint_policy_net15000.pth'
                self.policy_net.load_state_dict(torch.load(to_load))
                self.policy_net.eval()
            num_runs = 1
            self.strings_to_write_eval = []
            self.overall_interactions = []
            for i in range(0, num_runs):
                self.model.interactions_counter = 0
                self.model.EPISODE_REWARD = []
                self.model.reset()
                for t in itertools.count():
                    print(self.overall_interactions, self.model.interactions_counter)
                    # if done:
                    if self.model.END_EVAL:
                        self.model.END_EVAL = False
                        break
                    else:
                        self.model.step(num_run=i, auto=True, UAV_idx=UAV_idx, type=type,
                                        strings_to_write_eval=self.strings_to_write_eval,
                                        overall_interactions=self.overall_interactions)
                times.append(self.model.times)

            self.write_metrics_files(UAV_idx, type)

        print("--- TIMES ---")
        [print(times[idx]) for idx in range(0, max_agents)]

    def write_metrics_files(self, UAV_idx, type):
        # CHANGE STR(0) IF PARALLELIZATION IS NEEDED. PUT AS MANY NUMBERS AS CPUS ARE USED
        # (THIS AVOID COPYING PROJECTS UNNECESSARILY).
        with open(ROOT_RESULTS_DQN_INFERENCE_RESULTS + str(UAV_idx + 1) + 'UAV.txt',
                  'w') as f:
            f.writelines(self.strings_to_write_eval)
        with open(ROOT_RESULTS_DQN_INFERENCE_RESULTS + 'COUNTER_' + str(UAV_idx + 1) + 'UAV.txt',
                  'w') as f2:
            f2.writelines(self.overall_interactions)


class LSTMLearning:
    
    def __init__(self):
        pass
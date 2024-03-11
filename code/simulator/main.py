# python libraries
import h5py
import mesa
import sys

import torch

sys.path.append('../mesa_addons/')
sys.path.append('../NN/')
sys.path.append('../PCD/')

from Canvas_Grid_Visualization import CanvasGrid

# own python modules

import wildfire_model
import agents

import nn_learning_process
import horizon_evaluation

from common_fixed_variables import (INFERENCE, RNN, INTERFACE, WIDTH, HEIGHT, FIRE_COLORS, VEGETATION_COLORS,
                                    SMOKE_COLORS, N_ACTIONS, N_OBSERVATIONS, FUEL_UPPER_LIMIT, PROBABILITY_MAP,
                                    BLACK_AND_WHITE_COLORS, BURNING_FLAGS_MAP, HORIZON_EVAL, normalize_fuel_values)


# creates agent dictionary for rendering it on Canvas Gird
def agent_portrayal(agent):
    portrayal = {"Shape": "rect", "Filled": True, "h": 1, "w": 1}
    if PROBABILITY_MAP:
        if type(agent) is agents.Fire:
            idx = int(round(agent.get_prob(), 1) * 10)
            portrayal.update({"Color": BLACK_AND_WHITE_COLORS[idx], "Layer": 0})
    elif BURNING_FLAGS_MAP:
        if type(agent) is agents.Fire:
            if agent.is_burning():
                color = "Black"
            else:
                color = "White"
            portrayal.update({"Color": color, "Layer": 0})
    else:
        if type(agent) is agents.Fire:
            if agent.smoke.is_smoke_active():
                # idx = normalize_fuel_values(agent.smoke.get_dispelling_counter_value(),
                #                             agent.smoke.get_dispelling_counter_start_value())
                portrayal.update({"Color": SMOKE_COLORS[0], "Layer": 0})
            else:
                if agent.is_burning():
                    idx = normalize_fuel_values(agent.get_fuel(), FUEL_UPPER_LIMIT)
                    portrayal.update({"Color": FIRE_COLORS[idx], "Layer": 0})  # "#ff5d00" -> fire orange
                else:
                    idx = normalize_fuel_values(agent.get_fuel(), FUEL_UPPER_LIMIT)
                    portrayal.update({"Color": VEGETATION_COLORS[idx], "Layer": 0})
        elif type(agent) is agents.UAV:
            portrayal.update({"Color": "Black", "Layer": 1, "h": 0.8, "w": 0.8})
    return portrayal


def main():
    print('actions:', N_ACTIONS)
    print('observations:', N_OBSERVATIONS)

    open_dataset_LSTM = True
    if open_dataset_LSTM:
        # Open the HDF5 file
        with h5py.File('dataset_3UAV.hdf5', 'r') as f:
            # Access the dataset
            data = f['data'][:]
            # Convert to PyTorch tensor
            data_tensor = torch.tensor(data)

            size = data_tensor.size()
            seq_length = 10
            # inferred dimension based on 'seq_length', only if size[1] (num of simulations) is divisible
            # by 'seq_length', otherwise it must be changed
            reshaped = data_tensor.view(-1, seq_length, size[2], size[3])

            r_size = reshaped.size()
            reshaped_v2 = reshaped.view(r_size[0], r_size[1], -1)

    print('¡¡¡printing data!!!')
    print(data_tensor, data_tensor.shape)
    print('¡¡¡reshape!!!')
    print(reshaped, reshaped.shape)
    print('¡¡¡reshape v2!!!')
    print(reshaped_v2, reshaped_v2.shape)

    wf_model = wildfire_model.WildFireModel()
    deep_q_learning = nn_learning_process.DQNLearning(wildfiremodel=wf_model)

    if not INFERENCE:  # False = training
        deep_q_learning.train()
    else:  # True = inference (eval)
        if INTERFACE == 'Interface':
            grid = CanvasGrid(agent_portrayal, WIDTH, HEIGHT, 10 * WIDTH, 10 * HEIGHT)
            # initialize Modular server for mesa Python visualization
            server = mesa.visualization.ModularServer(wildfire_model.WildFireModel, [grid], "WildFire Model")
            server.port = 8521  # The default
            server.launch()
        else:
            if HORIZON_EVAL:
                horizon_eval = horizon_evaluation.HorizonEvaluation()
                horizon_eval.evaluation()
            else:
                print('Launched automatic evaluation, without interface ...')
                type = 'predictor' if RNN is True else 'entrenamiento'
                print(type)
                deep_q_learning.evaluation(type)


main()

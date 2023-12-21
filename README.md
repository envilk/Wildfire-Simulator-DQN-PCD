# Project overview

## Project description

This repository holds a Predictive Coordinate Descent (PCD) algorithm, and a Deep Q-Network (DQN) algorithm, in order to enable decentralized proactive self-adaptation in Smart Cyber-Physical Systems (sCPS). Concretely, these algorithm were tested in wildfire tracking adaptation scenarios, included in Unmanned Aerial Vehicles (UAV). For simulating UAV systems and different uncertain widlfire scenarios, Wildfire-UAVSim is used in this repository, which consists of a customizable wildfire tracking simulator that enables the evaluation of diverse adaptation strategies. For further description of the Wildfire-UAVSim simulator check [its repository](https://anonymous.4open.science/r/Wildfire-UAVSim-E758/).

## Project structure

The project contains the following structure:

📦project \
 ┣ 📂code \
 ┃ ┣ 📂DQN \
 ┃ ┃ ┣ 📜data_structures.py \
 ┃ ┃ ┣ 📜nn_definitions.py \
 ┃ ┃ ┗ 📜nn_learning_process.py \
 ┃ ┣ 📂PCD \
 ┃ ┃ ┗ 📜horizon_evaluation.py \
 ┃ ┣ 📂simulator \
 ┃ ┃ ┣ 📜agents.py \
 ┃ ┃ ┣ 📜common_fixed_variables.py \
 ┃ ┃ ┣ 📜main.py \
 ┃ ┃ ┗ 📜wildfire_model.py \
 ┃ ┣ 📂source_modified \
 ┃ ┃ ┣ 📜Canvas_Grid_Visualization.py \
 ┃ ┃ ┗ 📜Space_Grid_Setter_Getter.py \
 ┃ ┣ 📂statistics \
 ┃ ┃ ┣ 📜box_diagram_MR1_MR2.py \
 ┃ ┃ ┣ 📜box_diagram_MR1_MR2_degradations.py \
 ┃ ┃ ┣ 📜extract_data_for_boxplots.py \
 ┃ ┃ ┣ 📜extract_data_for_boxplots_degradations.py \
 ┃ ┃ ┗ 📜training_results.py \
 ┃ ┗ 📜paths.py \
 ┗ 📂results \
 ┃ ┣ 📂DQN \
 ┃ ┃ ┣ 📂inference_results \
 ┃ ┃ ┣ 📂training_checkpoints \
 ┃ ┃ ┗ 📂training_results \
 ┃ ┗ 📂PCD \
 ┃ ┃ ┣ 📂inference_grid_instants \
 ┃ ┃ ┗ 📂inference_results

### code folder

| File | Description |
|----------|----------|
| `data_structures.py` | This python file holds data structures for training Neural Networks used in this project. For the moment, only Replay Memory is used for training DQN |
| `nn_definitions.py` | This python file holds the logic for building Neural Networks structures. For the moment, only DQN is built |
| `nn_learning_process.py` | This python file holds the logic for training and evaluating Neural Networks used in this project. For the moment, only evaluation and training processes for DQN are developed |
| `horizon_evaluation.py` | This python file holds the logic for executing PCD algorithm evaluation process |
| `agents.py` | This python file holds the logic for managing elements such as Fire, Smoke, Wind and UAVs |
| `widlfire_model.py` | This python file holds the logic for managing the wildfire simulation, by utilizing elements from `agents.py` file |
| `main.py` | This python file allows to execute the wildfire simulation built in the rest of the files |
| `common_fixed_variables.py` | This python file holds the variables used to set the simulation execution configurations |
| `Canvas_Grid_Visualization.py` | This python file contains a Mesa class, modified for making UAV observation areas visible on the graphical web interface. It is not really necessary to change this file. |
| `Space_Grid_Setter_Getter.py` | This python file contains a Mesa class, modified for making a concrete functionality of PCD work properly. It is not really necessary to change this file |
| `box_diagram_MR1_MR2.py` | This python file contains the logic to show box plot statistics of the two defined metrics MR1 and MR2 |
| `box_diagram_MR1_MR2_degradations.py` | This python file is a pretty similar adaptation of `box_diagram_MR1_MR2.py`, but allowing to plot statistics of several PCD degradations |
| `extract_data_for_boxplots.py` | This python file contains the logic to extract data for plotting in `box_diagram_MR1_MR2.py` |
| `extract_data_for_boxplots_degradations.py` | This python file contains the logic to extract data for plotting in `box_diagram_MR1_MR2_degradations.py` |
| `training_results.py` | This python file contains the logic for plotting training metrics |
| `paths.py` | This python file define variables containing paths globally used in the project |

### results folder

| Directory | Description |
|----------|----------|
| `./DQN/inference_results/` | In this folder, MR1 and MR2 metrics results of the DQN performance are stored |
| `./DQN/training_checkpoints/` | In this folder, model checkpoints extracted from the training process are stored, in `.pth` file format |
| `./DQN/training_results/` | In this folder, training metrics results are stored |
| `./PCD/inference_grid_instants/` | In this folder, all grid time steps for each simulation of the PCD automatic evaluation process are stored, in `.png` format, for keeping track of how the grid evolved during different simulations |
| `./PCD/inference_results/` | In this folder, MR1 and MR2 metrics results of the PCD performance are stored |

# Installation setup

The installation setup for executing the project, is based on Pycharm Community Edition IDE. The installation process can be checked in the **Installation setup** section in [Wildfire-UAVSim repository](https://anonymous.4open.science/r/Wildfire-UAVSim-E758/). For properly installing everything needed, first follow referenced installation process, then complement each section with the described below (some section names are the same in purpose, to add information).

## Opening the project

For opening this project, instead of extracting Wildfire-UAVSim, extract the Wildfire-Simulator-DQN-PCD downloaded package in any folder, and then follow same steps as mentioned in the same section of Wildfire-UAVSim repository.

## Installing dependencies

As well as explained in Wildfire-UAVSim repository, some dependencies must be installed, keeping the same installation process. For this project, the list of dependencies grows:

<ul>
  <li>Mesa (v.1.2.1)</li>
  <li>numpy (v1.24.2)</li>
  <li>pandas (v2.0.3)</li>
  <li>plotly (v5.15.0)</li>
  <li>torch (v2.0.0)</li>
  <li>seaborn (v0.12.2)</li>
  <li>dash (v2.11.1)</li>
</ul>

Extra libraries not included in the list might be needed. In that case, follow Pycharm IDE installation recommendations when throwing dependency errors. This can be done by hovering the dependency errors in Pycharm.

# Execution of the project

Once installation set up is completed, `main.py` can be executed by selecting the file inside `./code/simulator/` folder, right mouse click, and clicking on `Run 'main'` (shortcut should be `Ctrl+Mayus+F10`).

## Project execution options

By executing `main.py` file, a list of different execution options should appear in Pycharm console:

```
------------ COMMON_FIXED_VARIABLES.PY ------------
--------------------- OPTIONS: --------------------
1. Train DQN
2. DQN automatic evaluation
3. PCD automatic evaluation
4. Random selection automatic evaluation (future RNN/LSTM/etc)
5. DQN interface evaluation
6. Random selection interface evaluation (future RNN/LSTM/etc)
SELECT AN OPTION (introduce an int number):
```

`1. Train DQN` Lorem ipsum ...

`2. DQN automatic evaluation` Lorem ipsum ...

`3. PCD automatic evaluation` Lorem ipsum ...

`4. Random selection automatic evaluation (future RNN/LSTM/etc)` Lorem ipsum ...

`5. DQN interface evaluation` Lorem ipsum ...

`6. Random selection interface evaluation (future RNN/LSTM/etc)` Lorem ipsum ...

## Statistics execution options

Lorem ipsum ...

For further explanation about the web graphical interface elements, and the common variables configuration for evaluation of different algorithms in several scenarios, check sections `Graphical interface functionalities` and `Common variables configuration` in Wildfire-UAVSim repository, respectively.

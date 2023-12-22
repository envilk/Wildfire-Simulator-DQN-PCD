# Project overview

## Project description

This repository holds a Predictive Coordinate Descent (PCD) algorithm, and a Deep Q-Network (DQN) algorithm, in order to enable decentralized proactive self-adaptation in Smart Cyber-Physical Systems (sCPS). Concretely, these algorithms were tested in wildfire tracking adaptation scenarios, included in Unmanned Aerial Vehicles (UAV). For simulating UAV systems and different uncertain wildfire scenarios, Wildfire-UAVSim is used in this repository, which consists of a customizable wildfire tracking simulator that enables the evaluation of diverse adaptation strategies. For further description of the Wildfire-UAVSim simulator check [its repository](https://anonymous.4open.science/r/Wildfire-UAVSim-E758/).

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
 ┃ ┣ 📂mesa_addons \
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

| Directory | Description |
|----------|----------|
| `./DQN/` | This directory includes python files for managing the logic of the DQN algorithm |
| `./PCD/` | This directory includes python files for managing the logic of the PCD algorithm |
| `./simulator/` | This directory includes python files for managing the simulator |
| `./mesa_addons/` | This directory includes python files for adding functionalities to some mesa classes |
| `./statistics/` | This directory includes python files for generating statistics about different executions of the project |

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
| `box_diagram_MR1_MR2_degradations.py` | This python file is a pretty similar adaptation of `box_diagram_MR1_MR2.py`, but allows to plot statistics of several PCD degradations |
| `extract_data_for_boxplots.py` | This python file contains the logic to extract data for plotting in `box_diagram_MR1_MR2.py` |
| `extract_data_for_boxplots_degradations.py` | This python file contains the logic to extract data for plotting in `box_diagram_MR1_MR2_degradations.py` |
| `training_results.py` | This python file contains the logic for plotting training metrics |
| `paths.py` | This python file defines variables containing paths globally used in the project |

### results folder

| Directory | Description |
|----------|----------|
| `./DQN/inference_results/` | In this folder, MR1 and MR2 metrics results of the DQN performance are stored |
| `./DQN/training_checkpoints/` | In this folder, model checkpoints extracted from the training process are stored, in `.pth` file format |
| `./DQN/training_results/` | In this folder, training metrics results are stored |
| `./PCD/inference_grid_instants/` | In this folder, all grid time steps for each simulation of the PCD automatic evaluation process are stored, in `.png` format, for keeping track of how the grid evolved during different simulations |
| `./PCD/inference_results/` | In this folder, MR1 and MR2 metrics results of the PCD performance are stored |

# Installation setup

In the following subsections, the installation process for executing the project will be explained.

## Installing Pycharm Community Edition IDE

The first step is to download and install Pycharm Community Edition IDE to easily run and set up the project and its dependencies. For Linux users (this project was initially tested on Ubuntu 22.04.2 LTS), they can use the `snap` command in cmd (pre-installed from Ubuntu 16.04 LTS and later) as a fast installation option. Users must execute the command `sudo snap install pycharm-community --classic` in cmd for installing Pycharm Community Edition.

Despite this project was initially tested on Ubuntu 22.04.2 LTS, it has been later tested on Windows and Mac too. For checking system requirements, and information about the installation process, please visit https://www.jetbrains.com/help/pycharm/installation-guide.html.

## Opening the project

First, extract the Wildfire-Simulator-DQN-PCD downloaded package in any folder. Second, open Pycharm by executing the command `pycharm-community` in cmd, or searching for the executable in the computer.
Then, the project window should be opened. Next, the user has to click on `Open`, select the extracted project folder, and click `OK`. A window should appear to select between light editor, and project editor.
Select project editor. For openning the project more times, repeat same process.

## Installing dependencies

Once the project is opened, some dependencies are necessary. To install them, first go to `Settings > Project > Python Interpreter`, then select the desired Python interpreter
for executing the project. As default `/usr/bin/python3.10` should appear in the `Python Interpreter:` tab, which already contains some default dependencies if Ubuntu 22.04.2 LTS is installed. For other Python interpreters,
other dependencies may need to be installed. On the same Pycharm configuration window, click on `+` icon, and search for the following dependencies (the user should specify the same version as the one used when developing the project. Version can be specified by clicking on the "Specify version" checkbox):

<ul>
  <li>Mesa (v.1.2.1)</li>
  <li>numpy (v1.24.2)</li>
  <li>pandas (v2.0.3)</li>
  <li>plotly (v5.15.0)</li>
  <li>torch (v2.0.0)</li>
  <li>seaborn (v0.12.2)</li>
  <li>dash (v2.11.1)</li>
</ul>

Extra dependencies not included in the list might be needed. In that case, follow Pycharm IDE installation recommendations when throwing dependency errors. This can be done by hovering the dependency errors in Pycharm.

# Execution of the project

Once the installation setup is completed, `main.py` can be executed by selecting the file inside `./code/simulator/` folder, right mouse click, and clicking on `Run 'main'` (shortcut should be `Ctrl+Shift+F10`).

## Graphical interface functionalities

When executing the project as explained above, a web page hosted in http://127.0.0.1:8521/ should appear in user's default browser. Port can be modified in `main.py` file if user has the default one already busy.

The relevant graphical interface elements are shown in the following table:

| Element | Description |
|----------|----------|
| `Grid` | The grid with generated cells, with vegetation, fire, smoke, and UAVs, can be seen in the center of the screen. |
| `Start button` | The start button allows to run the simulation without stopping. |
| `Step button` | The step button allows to execute one time step at a time. |
| `Reset button` | The reset button allows to execute the `reset()` method, inherited and overwritten from Mesa framework class `mesa.Model`, into WildFireModel class, inside `widlfire_model.py` file. |
| `Frames per second` | It is a slider that allows to set the frames per second (FPS) velocity for the graphical visualization of the simulation execution. Each frame corresponds to one time step. Its range goes from 1 to 20 FPS, taking into account that, counterintuitively, 0 FPS sets the fastest FPS velocity. One reason why the simulation might seem not be playing fluently could be the setting of the `FIRE_SPREAD_SPEED` variable referenced below. |
| `Current step counter` | Indicates the current time step of the simulation. |

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

`1. Train DQN` This option allows to train DQN algorithm. Concretely, model checkpoints will appear in `.results/DQN/training_checkpoints/` directory. Take into account that when executing options for doing inference with DQN, model checkpoints in this directory will be used (**by default, pre-loaded model checkpoints for testing with each UAV amount, are held in the repository. Take into account that these checkpoints were obtained from training DQN on normal conditions scenarios**). Also, training metrics results will appear in `.results/DQN/training_results/` directory, in the shape of files such as `*UAV_training_results.txt`, for each corresponding UAV amount.

`2. DQN automatic evaluation` This option will execute an automatic process for evaluating DQN performance in a certain wildfire scenario. Concretely, results will appear in `.results/DQN/inference_results/` directory, in the shape of folders such as `*UAV_run_rewards` (including statistics plotting each obtained reward concerning its corresponding time step, for each simulation), files such as `*UAV.txt` (each row is the result of obtaining effective wildfire monitoring metric MR1, on each simulation), and such as `COUNTER_*UAV.txt` (each row is the result of obtaining collision risk avoidance metric MR2, on each simulation), for each UAV amount, respectively.

`3. PCD automatic evaluation` This option will execute an automatic process for evaluating PCD performance in a certain wildfire scenario. Concretely, results will appear in `.results/PCD/inference_results/` directory, in the shape of folders such as `*UAV_run_rewards` (including statistics plotting each obtained reward concerning its corresponding time step, for each simulation), files such as `*UAV.txt` (each row is the result of obtaining effective wildfire monitoring metric MR1, on each simulation), and such as `COUNTER_*UAV.txt` (each row is the result of obtaining collision risk avoidance metric MR2, on each simulation), for each UAV amount, respectively. Also, for PCD automatic evaluation, each grid time step of each simulation will be plotted in `.results/PCD/inference_grid_instants/` (since it has no graphical web interface evaluation).

`4. Random selection automatic evaluation (future RNN/LSTM/etc)` This option will execute an automatic process for evaluating random selection performance in a certain wildfire scenario. Concretely, results will appear in `.results/DQN/inference_results/` directory (**random selection does not have its directory**), in the shape of folders such as `*UAV_run_rewards` (including statistics plotting each obtained reward concerning its corresponding time step, for each simulation), files such as `*UAV.txt` (each row is the result of obtaining effective wildfire monitoring metric on each simulation), and such as `COUNTER_*UAV.txt` (each row is the result of obtaining collision risk avoidance metric on each simulation), for each UAV amount, respectively.

`5. DQN interface evaluation` This option will execute the graphical web interface for evaluating DQN algorithm. Also, statistics will be plotted in the shape of images, such as `*_EVAL_drones_pyfoo.svg` for the concrete UAV amount. Statistics will plot each obtained reward concerning its corresponding time step, for each simulation.

`6. Random selection interface evaluation (future RNN/LSTM/etc)` This option will execute the graphical web interface for evaluating random selection. Also, statistics will be plotted in the shape of images, such as `*_EVAL_drones_pyfoo.svg` for the concrete UAV amount. Statistics will plot each obtained reward concerning its corresponding time step, for each simulation.

## Statistics execution options

Statistics are shown through Python files contained in `./DQN/statistics/` directory, namely:

`extract_data_for_boxplots.py` This file allows to extract data for both metrics, MR1 and MR2, to quantify DQN and PCD performances. An output example from Pycharm console when executing the file, can be seen below.

```
[[28.044979095458984], [31.39619731903076], [30.931952158610027]]
[[30.826993942260742], [32.96366786956787], [32.67935434977213]]
[[0.0], [21.0], [37.0]]
[[0.0], [12.0], [22.0]]
```

Specifically, the first two lists correspond to DQN and PCD MR1 metrics, respectively. The two following lists correspond to DQN and PCD MR2 metrics, respectively. There should be three lists inside each mentioned list, to test each UAV amount.

`box_diagram_MR1_MR2.py` This file allows to save a box plot in `.pdf` format, to compare DQN and PCD performance, based on MR1 and MR2 metrics. First, to show the box plot, data extracted with `extract_data_for_boxplots.py` must be manually set as input. The correspondences (assigning values to variables) between shown lists and box plot variables can be seen in the table below (**example values from `extract_data_for_boxplots.py` are the same as the ones shown above**):

| Values from `extract_data_for_boxplots.py` | Variables from `box_diagram_MR1_MR2.py` |
|----------|----------|
| `[[28.044979095458984], [31.39619731903076], [30.931952158610027]]` | DQN_MR1_data |
| `[[30.826993942260742], [32.96366786956787], [32.67935434977213]]` | PCD_MR1_data |
| `[[0.0], [21.0], [37.0]]` | DQN_MR2_data |
| `[[0.0], [12.0], [22.0]]` | PCD_MR2_data |

`extract_data_for_boxplots_degradations.py` This file allows to extract data for both metrics, MR1 and MR2, to quantify PCD performance when it is degraded with three degradation intensities (LOW, MEDIUM, HIGH). For extracting the data, several steps must be manually developed before executing the file. First, the project folder must be copied three times, renamed with names `Wildfire-Simulator-DQN-PCD-A` (LOW intensity), `Wildfire-Simulator-DQN-PCD-B` (MEDIUM intensity), and `Wildfire-Simulator-DQN-PCD-C` (HIGH intensity), respectively. Then, tweak degradation variables from each project to fit each intensity grade (the user can use default intensity values, then set NOISE = True, and tweak OMEGA variable assignation with its corresponding value in each project), and run each project's automatic evaluation process. Lastly, execute `extract_data_for_boxplots_degradations.py` from any copied project. Once this process has finished, the user can rename the project with its default name. An output example from Pycharm console when executing the file, can be seen below.

```
[[29.826993942260742], [30.06366786956787], [27.67935434977213]]
[[22.126993942260742], [23.86366786956787], [23.47935434977213]]
[[20.826993942260742], [22.96366786956787], [22.27935434977213]]
[[0.0], [9.0], [34.0]]
[[0.0], [14.0], [60.0]]
[[0.0], [18.0], [78.0]]
```

Specifically, the first three lists correspond to PCD MR1 metrics, with LOW, MEDIUM, and HIGH grades of degradation, respectively. The three following lists correspond to PCD MR2 metrics, with LOW, MEDIUM, and HIGH grades of degradation, respectively. There should be three lists inside each mentioned list, to test each UAV amount.

`box_diagram_MR1_MR2_degradations.py` This file allows to save a box plot in `.pdf` format, to compare DQN and PCD degraded performances, based on MR1 and MR2 metrics. First, to show the box plot, data extracted with `extract_data_for_boxplots_degradations.py` must be manually set as input. The correspondences (assigning values to variables) between shown lists and box plot variables can be seen in the table below (**example values from `extract_data_for_boxplots_degradations.py` are the same as the ones shown above, as well as DQN examples from `extract_data_for_boxplots.py`**):

| File | Values | Variables from `box_diagram_MR1_MR2_degradations.py` |
|----------|----------|----------|
| `extract_data_for_boxplots.py` | `[[28.044979095458984], [31.39619731903076], [30.931952158610027]]` | DQN_MR1 |
| `extract_data_for_boxplots_degradations.py` | `[[29.826993942260742], [30.06366786956787], [27.67935434977213]]` | LOW_MR1 |
| `extract_data_for_boxplots_degradations.py` | `[[22.126993942260742], [23.86366786956787], [23.47935434977213]]` | MEDIUM_MR1 |
| `extract_data_for_boxplots_degradations.py` | `[[20.826993942260742], [22.96366786956787], [22.27935434977213]]` | HIGH_MR1 |
| `extract_data_for_boxplots.py` | `[[0.0], [12.0], [36.0]]` | DQN_MR2 |
| `extract_data_for_boxplots_degradations.py` | `[[0.0], [9.0], [34.0]]` | LOW_MR2 |
| `extract_data_for_boxplots_degradations.py` | `[[0.0], [14.0], [60.0]]` | MEDIUM_MR2 |
| `extract_data_for_boxplots_degradations.py` | `[[0.0], [18.0], [78.0]]` | HIGH_MR2 |

`training_results.py` Execute this Python file whenever the DQN training process has finished, and `*UAV_training_results.txt` files are ready in the folder `.results/DQN/training_results/`.

# Annex: Common variables configuration

Global variables are used in the project to configure different simulation executions. In the next subsections several global variables descriptions are shown, as well as many configuration examples for execution.

## Variables description

### Forest area

| Variable name | Description |
|----------|----------|
| `BATCH_SIZE` | It establishes how long the simulation will run, in number of time steps |
| `WIDTH`, `HEIGHT` | They set the grid size (forest area size) in cells |
| `BURNING_RATE` | It sets the fuel decay speed in terms of time steps |
| `FIRE_SPREAD_SPEED` | It sets how fast fire spreads to other cells, in terms of time steps |
| `FUEL_UPPER_LIMIT`, `FUEL_BOTTOM_LIMIT` | They establish the maximum and minimum amount of burnable fuel present in each cell, respectively |
| `DENSITY_PROB` | It is a value in the range `[0, 1]` that establishes the percentage of the grid covered by vegetation |

### Wind

| Variable name | Description |
|----------|----------|
| `ACTIVATE_WIND` | It sets whether the fire spread is influenced by wind |
| `FIXED_WIND` | If it is active, then wind blows in the direction set by `WIND_DIRECTION`. If it is not, it means wind blows two directions, specified by `FIRST_DIR` and `SECOND_DIR`. Since wind can blow a direction stronger than the other one, `FIRST_DIR_PROB` establishes the wind first direction’s predominance |
| `PROBABILITY_MAP` | If it is active, the probability of the fire to spread to each cell at all times can be visualized |
| `MU` | It sets how strong wind blows with a value in the range `[0, 1]` |

### Smoke

| Variable name | Description |
|----------|----------|
| `ACTIVATE_SMOKE` | It sets whether smoke will be part of the simulation |
| `SMOKE_PRE_DISPELLING_COUNTER` | It establishes how fast smoke appears after fire starts in a cell |

### UAV

| Variable name | Description |
|----------|----------|
| `NUM_AGENTS` | It establishes the amount of UAVs that will fly over the forest area (zero indicates the simulator will simulate only the wildfire spread) |
| `N_ACTIONS` | Specifies the number of possible actions each UAV can take when deciding on a move, which by default is set as `[north, east, west, south]` |
| `UAV_OBSERVATION_RADIUS` | It sets the observation radius—technically it is not really a radius, since observed areas have square shapes |
| `SECURITY_DISTANCE` | It establishes the minimum distance that UAVs should be separated from each other for avoiding collisions |

## Configuration examples

Six default examples of how different variables can be configured to develop distinct scenarios, can be seen below. All scenarios shown are captured in `time step = 20`, in different time steps scenarios might look different.

### Common default variables

Before showing the examples, this section compiles all variables that can be set in common with all examples. The variables that were not mentioned can be set to their default value.

| Variable name | Value |
|----------|----------|
| `BATCH_SIZE` | 90 |
| `WIDTH` | 50 |
| `HEIGHT` | 50 |
| `BURNING_RATE` | 1 |
| `FIRE_SPREAD_SPEED` | 2 |
| `FUEL_UPPER_LIMIT` | 10 |
| `FUEL_BOTTOM_LIMIT` | 7 |
| `DENSITY_PROB` | 1 |

### Normal conditions (no smoke, no wind, no UAV)

A scenario with no wind, smoke, or UAV, should appear.

| Variable name | Value |
|----------|----------|
| `NUM_AGENTS` | 0 |
| `ACTIVATE_WIND` | False |
| `ACTIVATE_SMOKE` | False |
| `PROBABILITY_MAP` | False |

### Windy conditions (no smoke, wind, no UAV)

Concretely, a scenario with two weak wind components should appear, first with 50% of south component, and a second west component with 50%. In this scenario, neither smoke nor UAV should appear.

| Variable name | Value |
|----------|----------|
| `NUM_AGENTS` | 0 |
| `ACTIVATE_WIND` | True |
| `ACTIVATE_SMOKE` | False |
| `PROBABILITY_MAP` | False |
| `FIXED_WIND` | False |
| `WIND_DIRECTION` | 'south' |
| `FIRST_DIR` | 'south' |
| `SECOND_DIR` | 'west' |
| `FIRST_DIR_PROB` | 0.5 |
| `MU` | 0.5 |

### Windy and partial observability conditions (smoke, wind, no UAV)

A scenario with strong windy conditions, blowing east, and late short-lasting smoke should appear. Remember that, since the dispelling counter for smoke is set in `Smoke` class by default, inside `agents.py` file, changes should be done to the `self.dispelling_counter_start_value` variable, inside `__init()__` method (`Smoke` class). Keep also in mind that `self.dispelling_counter_start_value + SMOKE_PRE_DISPELLING_COUNTER` should be greater than the amount of fuel assigned to each cell (for taking fewer risks, compare to `FUEL_UPPER_LIMIT`, which is the maximum possible amount of fuel of each cell), to avoid situations in which smoke dissipates before the end of the cell’s burning process.

| Variable name | Value |
|----------|----------|
| `NUM_AGENTS` | 0 |
| `ACTIVATE_WIND` | True |
| `ACTIVATE_SMOKE` | True |
| `PROBABILITY_MAP` | False |
| `FIXED_WIND` | True |
| `WIND_DIRECTION` | 'east' |
| `MU` | 0.95 |
| `SMOKE_PRE_DISPELLING_COUNTER` | 7 |
| `self.dispelling_counter_start_value` | 4 |

### 2 UAV with small partial areas (normal conditions)

A scenario with 2 UAV having small partial areas in normal conditions should appear.

| Variable name | Value |
|----------|----------|
| `NUM_AGENTS` | 2 |
| `ACTIVATE_WIND` | False |
| `ACTIVATE_SMOKE` | False |
| `PROBABILITY_MAP` | False |
| `UAV_OBSERVATION_RADIUS` | 3 |

### 3 UAV with big partial areas (smoke, no wind)

A scenario with 3 UAV having big partial areas, with fast long-lasting smoke, should appear.

| Variable name | Value |
|----------|----------|
| `NUM_AGENTS` | 3 |
| `ACTIVATE_WIND` | False |
| `ACTIVATE_SMOKE` | True |
| `PROBABILITY_MAP` | False |
| `SMOKE_PRE_DISPELLING_COUNTER` | 2 |
| `self.dispelling_counter_start_value` | 9 |
| `UAV_OBSERVATION_RADIUS` | 12 |

### Probability map

A scenario with normal conditions should appear. Keep in mind that changing wind conditions will affect the visualized probabilities. Also, remember to set 0 UAV when showing the fire probability map.

| Variable name | Value |
|----------|----------|
| `NUM_AGENTS` | 0 |
| `ACTIVATE_WIND` | False |
| `ACTIVATE_SMOKE` | False |
| `PROBABILITY_MAP` | True |

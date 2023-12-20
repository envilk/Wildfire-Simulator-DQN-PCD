# python libraries
import copy

import mesa
import random
import functools

# own python modules

from common_fixed_variables import (FIRE_SPREAD_SPEED, BURNING_RATE, FUEL_BOTTOM_LIMIT,
                                    FUEL_UPPER_LIMIT, UAV_OBSERVATION_RADIUS, ACTIVATE_SMOKE,
                                    ACTIVATE_WIND, MU, SYSTEM_RANDOM, FIXED_WIND, euclidean_distance)


class Fire(mesa.Agent):

    def __init__(self, unique_id, model, burning=False):
        super().__init__(unique_id, model)
        self.fuel = random.randint(FUEL_BOTTOM_LIMIT, FUEL_UPPER_LIMIT)
        self.burning = burning
        self.next_burning_state = None
        self.moore = True
        self.radius = 3
        self.selected_dir = 0
        self.steps_counter = 0
        self.cell_prob = 0.0

        # fire properties

        # smoke
        self.smoke = Smoke(fire_cell_fuel=self.fuel)

    def is_burning(self):
        return self.burning

    def get_fuel(self):
        return round(self.fuel)

    def get_prob(self):
        return self.cell_prob

    def probability_of_fire(self):
        probs = []
        if self.fuel > 0:
            adjacent_cells = self.model.grid.get_neighborhood(
                self.pos, moore=self.moore, include_center=False, radius=self.radius
            )
            for adjacent in adjacent_cells:
                agents_in_adjacent = self.model.grid.get_cell_list_contents([adjacent])
                for agent in agents_in_adjacent:
                    if type(agent) is Fire:
                        adjacent_burning = 1 if agent.is_burning() else 0
                        aux_prob = self.distance_rate(self.pos, adjacent, self.radius) * adjacent_burning
                        if ACTIVATE_WIND and (adjacent_burning == 1):
                            aux_prob = self.model.wind.apply_wind(aux_prob, self.pos, agent.pos)
                        probs.append(1 - aux_prob)
            if len(probs) == 0:  # probably due to a low tree density in the wildfire simulation
                P = 0
            else:
                P = 1 - functools.reduce(lambda a, b: a * b, probs)
        else:
            P = 0
        return P

    def distance_rate(self, s, s_, distance_limit):
        m_d = euclidean_distance(s[0], s[1], s_[0], s_[1])
        result = 0
        if m_d <= distance_limit:
            result = m_d ** -2.0
        return result

    def step(self):
        self.steps_counter += 1
        # make fire spread slower
        if self.steps_counter % FIRE_SPREAD_SPEED == 0:
            # if self.steps_counter == 26: # to model how the wind can suddenly change direction
            #     self.model.wind.wind_direction = 'south'
            self.cell_prob = self.probability_of_fire()
            generated = random.random()
            if generated < self.cell_prob:
                self.next_burning_state = True
            else:
                self.next_burning_state = False
            if self.burning and self.fuel > 0:
                self.fuel = self.fuel - BURNING_RATE
            if ACTIVATE_SMOKE:
                self.smoke.smoke_step(self.burning)

    def advance(self):
        # make fire spread slower
        if self.steps_counter % FIRE_SPREAD_SPEED == 0:
            self.burning = self.next_burning_state

    # FIXME: needed methods for HORIZON EVALUATION
    def get_everything(self):
        return (self.unique_id, self.model, self.fuel, self.burning, self.next_burning_state, self.moore,
                self.radius, self.selected_dir, self.steps_counter, self.pos, self.cell_prob, self.smoke)

    def set_everything(self, _unique_id, _model, _fuel, _burning, _next_burning_state, _moore,
                       _radius, _selected_dir, _steps_counter, _pos, _cell_prob, _smoke):
        self.unique_id = copy.deepcopy(_unique_id)
        self.model = _model
        self.fuel = copy.deepcopy(_fuel)
        self.burning = copy.deepcopy(_burning)
        self.next_burning_state = copy.deepcopy(_next_burning_state)
        self.moore = copy.deepcopy(_moore)
        self.radius = copy.deepcopy(_radius)
        self.selected_dir = copy.deepcopy(_selected_dir)
        self.steps_counter = copy.deepcopy(_steps_counter)
        self.pos = copy.deepcopy(_pos)
        self.cell_prob = copy.deepcopy(_cell_prob)
        self.smoke = copy.deepcopy(_smoke)


class Smoke():

    def __init__(self, fire_cell_fuel):
        self.smoke = False # activated when fire ignites
        self.dispelling_counter_start_value = fire_cell_fuel
        self.dispelling_lower_bound_start_value = 2  # TODO only to start with some value, subjective criteria
        self.dispelling_lower_bound = self.dispelling_lower_bound_start_value
        self.dispelling_counter = self.dispelling_counter_start_value

    def get_dispelling_counter_value(self):
        return self.dispelling_counter

    def get_dispelling_counter_start_value(self):
        return self.dispelling_counter_start_value

    def is_smoke_active(self):
        return self.smoke

    def subtract_dispelling_counter(self):
        self.dispelling_counter -= 1

    def smoke_step(self, burning):
        # if not burning and not self.smoke: pass
        if not self.smoke and self.dispelling_counter == self.dispelling_counter_start_value:
            if ((burning and self.dispelling_lower_bound == self.dispelling_lower_bound_start_value) or
                    (0 < self.dispelling_lower_bound < self.dispelling_lower_bound_start_value)):
                self.dispelling_lower_bound -= 1
            elif self.dispelling_lower_bound == 0:
                self.smoke = True
        elif self.smoke:
            if 0 < self.dispelling_counter <= self.dispelling_counter_start_value:
                self.subtract_dispelling_counter()
            elif self.dispelling_counter == 0:
                self.smoke = False


class Wind():

    def __init__(self):
        self.wind_direction = 'south'
        if not FIXED_WIND:
            # Possible mixed wind directions: NW, NE, SW, SE"
            self.first_dir = 'south'  # Introduce first wind direction (north, south, east, west):
            self.first_dir_prob = 0.8  # "Introduce first wind probability [0, 1]
            self.second_dir = 'east'  # Introduce second wind direction (probability calculated based on first one),
            # (North, South, East, West)

    def change_direction(self):
        if SYSTEM_RANDOM.random() < self.first_dir_prob:
            self.wind_direction = self.first_dir
        else:
            self.wind_direction = self.second_dir

    def apply_wind(self, aux_prob, relative_center_pos, adjacent_pos):
        if not FIXED_WIND:
            self.change_direction()
            print("Wind: ", self.wind_direction)
        if self.is_on_wind_direction(relative_center_pos, adjacent_pos):
            aux_prob = aux_prob + (MU * (1 - aux_prob))  # part of 1 I- 'aux_prob' probability is added, depending on mu
        else:
            aux_prob = aux_prob - (MU * aux_prob)   # part of 'aux_prob' probability is removed, depending on mu
        return aux_prob

    def is_on_wind_direction(self, relative_center_pos, adjacent_pos):
        on_wind_direction = False
        if self.wind_direction == 'east':
            if (relative_center_pos[0] > adjacent_pos[0]) and (relative_center_pos[1] == adjacent_pos[1]):
                on_wind_direction = True
        elif self.wind_direction == 'west':
            if (relative_center_pos[0] < adjacent_pos[0]) and (relative_center_pos[1] == adjacent_pos[1]):
                on_wind_direction = True
        elif self.wind_direction == 'north':
            if (relative_center_pos[1] > adjacent_pos[1]) and (relative_center_pos[0] == adjacent_pos[0]):
                on_wind_direction = True
        elif self.wind_direction == 'south':
            if (relative_center_pos[1] < adjacent_pos[1]) and (relative_center_pos[0] == adjacent_pos[0]):
                on_wind_direction = True
        return on_wind_direction


class UAV(mesa.Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.moore = True
        self.selected_dir = 0

    def not_UAV_adjacent(self, pos):
        can_move = True
        agents_in_pos = self.model.grid.get_cell_list_contents([pos])
        for agent in agents_in_pos:
            if type(agent) is UAV:
                can_move = False
        return can_move

    def surrounding_states(self):
        reward = 0
        positions = []
        surrounding_states = []
        adjacent_cells = self.model.grid.get_neighborhood(
            self.pos, moore=self.moore, include_center=True, radius=UAV_OBSERVATION_RADIUS
        )
        for cell in adjacent_cells:
            agents = self.model.grid.get_cell_list_contents([cell])
            for agent in agents:
                if type(agent) is Fire:
                    # if there is smoke, state is considered to be, a cell that is "no burning" is equal to "no reward"
                    # state is used in -> ACTION POLICY SELECTION
                    if ACTIVATE_SMOKE and agent.smoke.is_smoke_active():
                        state = 0
                    else:
                        state = int(agent.is_burning() is True)
                    surrounding_states.append(state)
                    # reward is used in -> ACTION PREDICTOR SELECTION (and training of policy)
                    reward += state
                    positions.append(cell)
        return surrounding_states, reward, positions

    def move(self):
        # directions = [0, 1, 2, 3]  # right, down, left, up
        # self.selected_dir = random.choice(directions)
        move_x = [1, 0, -1, 0]
        move_y = [0, -1, 0, 1]
        moved = False

        pos_to_move = (self.pos[0] + move_x[self.selected_dir], self.pos[1] + move_y[self.selected_dir])
        if not self.model.grid.out_of_bounds(pos_to_move) and self.not_UAV_adjacent(pos_to_move):
            self.model.grid.move_agent(self, tuple(pos_to_move))
            moved = True

        return moved

    def advance(self):
        self.move()

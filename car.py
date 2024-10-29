#!/usr/bin/env python3
# coding=UTF-8

import numpy as np
import yaml
import casadi as ca
from acados_template import AcadosModel

class Car:
    def __init__(self, state=None, control=None, config_file="config.yaml"):
        # Load config file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # The state of car
        self.state_dim = 5
        self.control_dim = 2
        if state is not None:
            self.state = state  
        else:
            self.state = np.zeros(self.state_dim)
        if control is not None:
            self.control = control
        else:
            self.control = np.zeros(self.control_dim)

        # Define the constraints
        self.wheelbase = config['wheel_base']
        self.max_v = config['max_velocity']
        self.max_a = config['max_acceleration']
        self.max_delta = np.deg2rad(config['max_steering'])
        self.max_ddelta = np.deg2rad(config['max_rating'])
        
        self.min_v = -self.max_v
        self.min_a = -self.max_a
        self.min_delta = -self.max_delta
        self.min_ddelta = -self.max_ddelta

        # Initial path
        self.stamp = 0
        self.path = [np.concatenate([[self.stamp], self.state, self.control])]

    def f_dynamic(self, state:np.array, control:np.array):
        # Dynamic model with noise
        return np.array([
            state[3] * np.cos(state[2]),
            state[3] * np.sin(state[2]),
            state[3] * np.tan(state[4]) / self.wheelbase,
            control[0] + np.random.rand(),
            control[1] + np.random.rand()
        ])

    def update_state(self, control, dt):
        # Using Euler
        p = self.state
        self.state = p + self.f_dynamic(p, control) * dt

        self.stamp += dt
        self.path.append(np.concatenate([[self.stamp], self.state, self.control]))

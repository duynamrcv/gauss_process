import numpy as np
import casadi as ca

from car import Car

class Controller:
    def __init__(self, car: Car, n_nodes=20, time_step=0.1):
        self.car = car
        self.N = n_nodes
        self.dt = time_step
        self.x_dim = 5
        self.u_dim = 2

        self.opti = ca.Opti()
        self.opt_states = self.opti.variable(self.x_dim, self.N+1)
        self.opt_controls = self.opti.variable(self.u_dim, self.N)

        # Dynamic model
        f = lambda x_, u_: ca.vertcat(*[
            x_[3] * ca.cos(x_[2]),
            x_[3] * ca.sin(x_[2]),
            x_[3] * ca.tan(x_[4]) / self.car.wheelbase,
            u_[0],
            u_[1]
        ])

        # Initial condition
        self.opt_x_ref = self.opti.parameter(self.x_dim, self.N+1)
        self.opti.subject_to(self.opt_states[:,0] == self.opt_x_ref[:,0])
        for i in range(self.N):
            x_next = self.opt_states[:,i] + f(self.opt_states[:,i], self.opt_controls[:,i]) * self.dt
            self.opti.subject_to(self.opt_states[:,i+1] == x_next)

        # Weight matrices
        q_cost = np.diag([10, 10, 0.1, 0.05, 0.05])
        r_cost = np.diag([0.1, 0.1])

        # Cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[:,i] - self.opt_x_ref[:,i+1]
            obj = obj + ca.mtimes([state_error_.T, q_cost, state_error_]) \
                      + ca.mtimes([self.opt_controls[:,i].T, r_cost, self.opt_controls[:,i]])
        self.opti.minimize(obj)

        self.opti.subject_to(self.opti.bounded(self.car.min_v, self.opt_states[3,:], self.car.max_v))
        self.opti.subject_to(self.opti.bounded(self.car.min_delta, self.opt_states[4,:], self.car.max_delta))
        self.opti.subject_to(self.opti.bounded(self.car.min_a, self.opt_controls[0,:], self.car.max_a))
        self.opti.subject_to(self.opti.bounded(self.car.min_ddelta, self.opt_controls[1,:], self.car.max_ddelta))

        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0}

        self.opti.solver('ipopt', opts_setting)

    def compute_control_signal(self, x_ref):
        # Set parameter, here only update initial state of x (x0)
        self.opti.set_value(self.opt_x_ref, x_ref)

        sol = self.opti.solve()

        u = sol.value(self.opt_controls)
        # x = sol.value(self.opt_states)
        return u[:,0]
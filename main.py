import numpy as np
import math

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from car import Car

config_file = "car_model.yaml"

def generate_training_data(car:Car, num_sample):
    # Create sample data for training
    xs = np.zeros((num_sample, 1))
    ys = np.zeros((num_sample, 1))
    yaws = np.zeros((num_sample, 1))
    vs = np.sort((car.max_v - car.min_v) * np.random.rand(num_sample, 1), axis=0) + car.min_v
    accs = np.sort((car.max_a - car.min_a) * np.random.rand(num_sample, 1), axis=0) + car.min_a
    deltas = np.sort((car.max_delta - car.min_delta) * np.random.rand(num_sample, 1), axis=0) + car.min_delta
    ddeltas = np.sort((car.max_ddelta - car.min_ddelta) * np.random.rand(num_sample, 1), axis=0) + car.min_ddelta

    states = np.concatenate([xs, ys, yaws, vs, deltas], axis=1)
    controls = np.concatenate([accs, ddeltas], axis=1)
    X_train = np.concatenate([states, controls], axis=1)
    
    y_train = []
    for i in range(num_sample):
        state = states[i,:]
        control = controls[i,:]
        f = car.f_dynamic(state, control)
        y_train.append(f)
    y_train = np.array(y_train)
    return X_train, y_train

def training_phase(num_sample):
    car = Car(config_file=config_file)
    X_train, y_train = generate_training_data(car, num_sample)
    
    # Define the kernel (RBF kernel)
    kernel = 1. * RBF(length_scale=1.0)

    # Create a Gaussian Process Regressor with the defined kernel
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit the Gaussian Process model to the training data
    gp.fit(X_train, y_train)

if __name__ == "__main__":
    training_phase(num_sample=200)
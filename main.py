import pandas as pd
import numpy as np
import pickle
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from car import Car
from controller import Controller

config_file = "car_model.yaml"
ref_file = "reference.txt"
model_file = "gauss.pkl"
save_file = "data.pkl"
is_train = False
n_nodes = 20
dt = 0.1
num_points_search = 40

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

    # Save
    with open(model_file, 'wb') as f:
        pickle.dump(gp, f)

def load_trajectory(file_name):
    df = pd.read_csv(file_name, sep=',')
    ref = np.array(df)[:,:3]
    return ref

def normal_angle(theta):
    theta -= np.floor((theta + np.pi)/(2*np.pi))*2*np.pi
    return theta

def get_local_reference(reference, state, index, num_point):
    x_local = np.array([0,0,0,state[3],state[4]])
    local_reference = [x_local]
    for i in range(num_point):
        idx = index + i
        if idx >= reference.shape[0]:
            idx = reference.shape[0] - 1

        # Convert global to local
        dx = reference[idx,0] - state[0]
        dy = reference[idx,1] - state[1]
        x  = dx * np.cos(-state[2]) - dy * np.sin(-state[2])
        y  = dx * np.sin(-state[2]) + dy * np.cos(-state[2])
        yaw = normal_angle(reference[idx,2] - state[2])
        local_reference.append(np.array([x, y, yaw, 0, 0]))
    return np.array(local_reference).T

def find_nearest_index(reference, state, index, num_points):
    x = state[0]; y = state[1]
    cx = reference[index:index+num_points,0]
    cy = reference[index:index+num_points,1]
    dx = x - cx; dy = y - cy
    dist = np.hypot(dx, dy)
    index += np.argmin(dist)
    return index

def main():
    # Load model
    with open(model_file,'rb') as f:
        gp = pickle.load(f)
        print(gp.kernel_)

    ref = load_trajectory(ref_file)
    car = Car(state=np.array([ref[0,0], ref[0,1], ref[0,2], 0, 0]), config_file=config_file)
    controller = Controller(car, n_nodes, dt)
    
    times = []
    i = 0
    # while(total_time > car.stamp):
    while i < ref.shape[0]-1:
        x0 = car.state
        i = find_nearest_index(ref, x0, i, num_points_search)
        x_ref = get_local_reference(ref, x0, i, n_nodes)

        start = time.time()
        signal = controller.compute_control_signal(x_ref)
        times.append(time.time() - start)
        
        car.update_state(signal, dt)


    with open(save_file, 'wb') as file:
        path = np.array(car.path)
        ref = np.array(ref)
        times = np.array(times)
        print("Max processing time: {:.4f}s".format(times.max()))
        print("Min processing time: {:.4f}s".format(times.min()))
        print("Mean processing time: {:.4f}s".format(times.mean()))
        data = dict()
        data['path'] = path
        data['ref'] = ref
        data['times'] = times
        pickle.dump(data, file)

if __name__ == "__main__":
    if is_train:
        # Training Process
        training_phase(num_sample=200)
    else:
        main()
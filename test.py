#!/usr/bin/env python3

import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from gp.models import GaussianProcess
from gp.kernels import RBFKernel, WhiteNoiseKernel

def secret_function(x, noise=0.0):
    return torch.sin(x) + noise * torch.randn(x.shape)

def print_loss(gp):
    print("loss", gp.loss().detach().numpy())

def print_parameters(gp):
    for name, value in gp.named_parameters():
        print(name, value.detach().numpy())

def plot_variance(gp, x, y, title=None, std_factor=1.0):
    mu, std = gp(x, return_std=True)
    std *= std_factor
    
    mu = mu.detach().numpy()
    std = std.detach().numpy()
    
    x = x.numpy()
    y = y.detach().numpy()

    plt.figure(figsize=(12, 6))
    samples_plt, = plt.plot(
        X.numpy(), Y.numpy(),
        "bs", ms=4, label="Sampled points")
    y_plt, = plt.plot(x, y, "k--", label="Ground truth")
    mean_plt, = plt.plot(x, mu, "r", label="Estimate")
    std_plt = plt.gca().fill_between(
        x.flat, (mu - 3 * std).flat, (mu + 3 * std).flat,
        color="#dddddd", label="Three standard deviations")
    plt.axis([-4, 4, -2, 2])
    plt.title("Gaussian Process Estimate" if title is None else title)
    plt.legend(handles=[samples_plt, y_plt, mean_plt, std_plt])
    plt.show()

# Training data.
X = 10 * torch.rand(40, 1) - 4
X = torch.tensor(sorted(torch.cat([X] * 4))).reshape(-1, 1)
Y = secret_function(X, noise=1e-1)


# Construct GP.
k = RBFKernel() + WhiteNoiseKernel()
gp = GaussianProcess(k)
gp.set_data(X, Y)
gp.fit()

# Test data.
x = torch.linspace(-4, 4, 20).reshape(-1, 1)
y = secret_function(x)

plot_variance(gp, x, y)
print_loss(gp)
print_parameters(gp)
"""
Create, 11/14/2019

@author: Rui Meng
"""

import numpy as np 
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(22)
    n_input = 100
    n_val = 100
    X = np.random.rand(n_input, 1)
    Y = np.sin(2 * X) + 0.2 * np.cos(22 * X) + np.random.randn(n_input, 1) * 0.1
    X_validation = np.random.rand(n_val, 1)
    Y_validation = np.sin(2 * X_validation) + 0.2 * np.cos(22 * X_validation) + np.random.randn(n_val, 1) * 0.1
    Xs = np.linspace(0, 1, 100)[:, None]
    Fs = np.sin(2 * Xs) + 0.2 * np.cos(22 * Xs)

    with open("SIM_1D.pickle", "wb") as res:
        pickle.dump([X, Y, X_validation, Y_validation, Xs, Fs], res)

    # figure = plt.figure()
    # plt.scatter(X, Y, label="Observations")
    # plt.plot(Xs, Fs, label="True Data")
    # plt.show() 

    np.random.seed(22)
    n_input = 100
    n_val = 100
    X = np.concatenate([np.random.rand(int(n_input/2), 1)*0.3, np.random.rand(int(n_input/2), 1)*0.3+0.7])
    Y = np.sin(2 * X) + 0.2 * np.cos(22 * X) + np.random.randn(n_input, 1) * 0.1
    X_validation = np.random.rand(n_val, 1)
    Y_validation = np.sin(2 * X_validation) + 0.2 * np.cos(22 * X_validation) + np.random.randn(n_val, 1) * 0.1
    Xs = np.linspace(0, 1, 100)[:, None]
    Fs = np.sin(2 * Xs) + 0.2 * np.cos(22 * Xs)
    
    with open("SIM_1D_clustered.pickle", "wb") as res:
        pickle.dump([X, Y, X_validation, Y_validation, Xs, Fs], res)

    # figure = plt.figure()
    # plt.scatter(X, Y, label="Observations")
    # plt.plot(Xs, Fs, label="True Data")
    # plt.show() 

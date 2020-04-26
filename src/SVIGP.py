import numpy as np
import pandas as pd
import time
import sys
import pickle
from scipy.stats import norm
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt 

sys.path.append("..")
sys.path.append("../..")

from GPflow import gpflow
from GPflow.gpflow.likelihoods import Gaussian
from GPflow.gpflow import kernels
from GPflow.gpflow.models.svgp import SVGP
from GPflow.gpflow.training import ScipyOptimizer, NatGradOptimizer
from Utils.S_Plot import *


def batch_assess(model, assess_model, X, F):
    n_batches = max(int(X.shape[0]/1000.), 1)
    lik, sq_diff, ms, vs = [], [], [], []
    for X_batch, F_batch in zip(np.array_split(X, n_batches), np.array_split(F, n_batches)):
        l, sq, m, v = assess_model(model, X_batch, F_batch)
        lik.append(l)
        sq_diff.append(sq)
        ms.append(m)
        vs.append(v)
    lik = np.concatenate(lik, 0)
    sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
    ms = np.concatenate(ms, 0)
    vs = np.concatenate(vs, 0)
    return np.average(lik), np.average(sq_diff)**0.5, ms, vs


def assess_single_layer(model, X_batch, F_batch):
    m, v = model.predict_f(X_batch)
    lik = np.sum(norm.logpdf(F_batch, loc=m, scale=v**0.5),  1)
    sq_diff = (m - F_batch)**2
    return lik, sq_diff, m, v


from sklearn.model_selection import KFold
def cross_validation(X, Y, init_Z=np.linspace(0, 1, 10), init_hyper_parameters=np.array([-2,-2,-2]), lamb=1., kernel=kernels.Matern32(input_dim=1), n_fold = 5):
    kf = KFold(n_splits=n_fold)
    rmses = list()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]    
        kernel.variance = np.exp(init_hyper_parameters[1])
        kernel.lengthscales=np.exp(init_hyper_parameters[2])
        m = SVGP(X, Y.reshape([-1,1]), kernel, Gaussian(), init_Z.reshape([-1,1]), regularization_type=3, lamb=lamb)
        m.feature.set_trainable(True)
        m.likelihood.variance = np.exp(init_hyper_parameters[0])
        # Optimization
        m.compile()
        opt = ScipyOptimizer()
        opt.minimize(m, disp=False)
        # Prediction
        lik, rmse, m_grid, v_grid = batch_assess(m, assess_single_layer, X_test, y_test.reshape([-1,1]))
        rmses.append(rmse)
    return np.mean(np.array(rmses))

if __name__ == "__main__":
    with open("SIM_1D.pickle", "rb") as res:
        [X, Y, X_validation, Y_vailidation, Xs, Fs] = pickle.load(res)

    Fs = Fs.reshape(-1)
    n_IPs = 10
    init_Z = np.linspace(0, 1, n_IPs)
    init_hyper_parameters = np.array([-2, -2, -2])  # log_sigma2_err, log_sigma2, log_l
    kernel = kernels.Matern32(input_dim=1)

    # SVGP_optimized with hyper-parameters
    name = "1D_sim_SVGP_opt"
    ### Train hyper parameters
    kernel.variance = np.exp(init_hyper_parameters[1])
    kernel.lengthscales=np.exp(init_hyper_parameters[2])
    m = SVGP(X, Y.reshape([-1,1]), kernel, Gaussian(), init_Z.reshape([-1,1]))
    m.feature.set_trainable(False)
    m.likelihood.variance = np.exp(init_hyper_parameters[0])
    # Optimization
    m.compile()
    opt = ScipyOptimizer()
    opt.minimize(m, disp=False)
    # Prediction
    lik, rmse, m_grid, v_grid = batch_assess(m, assess_single_layer, Xs, Fs.reshape([-1,1]))
    print("0, lik: {:.4f}, rmse:{:.4f}".format(lik, rmse))

    # SVGP_optimized with hyper-parameters and IPs
    name = "1D_sim_SVGP_opt"
    ### Train hyper parameters
    kernel.variance = np.exp(init_hyper_parameters[1])
    kernel.lengthscales=np.exp(init_hyper_parameters[2])
    m = SVGP(X, Y.reshape([-1,1]), kernel, Gaussian(), init_Z.reshape([-1,1]))
    m.feature.set_trainable(True)
    m.likelihood.variance = np.exp(init_hyper_parameters[0])
    # Optimization
    m.compile()
    opt = ScipyOptimizer()
    opt.minimize(m, disp=False)
    # Prediction
    lik, rmse, m_grid, v_grid = batch_assess(m, assess_single_layer, Xs, Fs.reshape([-1,1]))
    print("1, lik: {:.4f}, rmse:{:.4f}".format(lik, rmse))
    
    # # Plot
    # low_CI = (m_grid - 1.96*v_grid**0.5).reshape(-1)
    # mid_CI = m_grid.reshape(-1)
    # upper_CI = (m_grid + 1.96*v_grid**0.5).reshape(-1)
    # sorted_x = Xs.reshape(-1)
    # x_label = " "
    # y_label = " "
    # title = None
    # save_dir=""
    # save_name= name + ".png"
    # Z = m.feature.Z.read_value().reshape(-1)
    # print(type(Z), Z)
    # # lineplotCI(Xs, Fs, sorted_x, low_CI, mid_CI, upper_CI, x_label, y_label, title, save_dir = save_dir, save_name = save_name, Z = Z)
    # lineplotCI0(X, Y, Xs, Fs, sorted_x, low_CI, mid_CI, upper_CI, x_label, y_label, title, save_dir = save_dir, save_name = save_name, Z = Z)


    # # SVGP_proposed
    # name = "1D_sim_SVGP_proposed"
    # ### Select optimal lambda using validation dataset
    # lamb_set = np.arange(10)
    # score = list()
    # for lamb in lamb_set:     
    #     ### Train hyper parameters
    #     k = kernels.Matern32(input_dim=1)
    #     k.variance = np.exp(init_hyper_parameters[1])
    #     k.lengthscales=np.exp(init_hyper_parameters[2])

    #     m = SVGP(X, Y.reshape([-1,1]), k, Gaussian(), init_Z.reshape([-1,1]), regularization_type=3, lamb=lamb)
    #     m.feature.set_trainable(True)
    #     m.likelihood.variance = np.exp(init_hyper_parameters[0])
    #     # print(m.as_pandas_table())

    #     # Optimization
    #     m.compile()
    #     opt = ScipyOptimizer()
    #     opt.minimize(m, disp=False)

    #     # Prediction
    #     lik_validation, rmse_validation, m_validation, v_validation = batch_assess(m, assess_single_layer, X_validation, Y_vailidation)
    #     score.append(rmse_validation)
    # score = np.array(score)
    # opt_lamb = lamb_set[np.argmin(score)]
    # print(score)
    # print("best lambda: {}".format(opt_lamb))

    # Select optimal lambda using cross validation
    lamb_set = np.arange(10)
    rmses = list()
    for lamb in lamb_set:
        print(lamb)
        rmse = cross_validation(X, Y, init_Z, init_hyper_parameters, kernel=kernel, lamb = lamb)
        rmses.append(rmse)
    opt_lamb = lamb_set[np.argmin(rmses)]
    print("best lambda: {}".format(opt_lamb))

    ### Train hyper parameters
    kernel.variance = np.exp(init_hyper_parameters[1])
    kernel.lengthscales=np.exp(init_hyper_parameters[2])
    m = SVGP(X, Y.reshape([-1,1]), kernel, Gaussian(), init_Z.reshape([-1,1]), regularization_type=3, lamb=opt_lamb)
    m.feature.set_trainable(True)
    m.likelihood.variance = np.exp(init_hyper_parameters[0])
    # print(m.as_pandas_table())
    # Optimization
    m.compile()
    opt = ScipyOptimizer()
    opt.minimize(m, disp=False)
    # Prediction
    lik_test, rmse_test, m_test, v_test = batch_assess(m, assess_single_layer, Xs, Fs.reshape([-1,1]))
    print("2, lik_test: {:.4f}, rmse_test:{:.4f}".format(lik, rmse))
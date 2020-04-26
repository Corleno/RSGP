import numpy as np 
import pickle
from scipy.spatial import distance
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def RBF_cov(X0, X1=None, sigma2=1., l=1.):
    if X1 is None:
        X1 = X0
    if len(X0.shape) == 1:
        X0 = X0.reshape([-1,1])
    if len(X1.shape) == 1:
        X1 = X1.reshape([-1,1])    
    dist = distance.cdist(X0/l, X1/l)
    return sigma2 * np.exp(-0.5*dist**2)


def Matern32_cov(X0, X1=None, sigma2=1., l=1.):
    if X1 is None:
        X1 = X0
    if len(X0.shape) == 1:
        X0 = X0.reshape([-1,1])
    if len(X1.shape) == 1:
        X1 = X1.reshape([-1,1])    
    dist = distance.cdist(X0/l, X1/l)
    return sigma2 * (1 + np.sqrt(3)*dist) * np.exp(-np.sqrt(3)*dist)


# compute the logN(Y|0, S = K_12K_22^-1K_21 + sigma^2I)
def log_Sparse_Guassian(Y, K12, K22, K21, sigma2):
    N, M = K12.shape
    # compute log of determinant of S
    A = K22 + K21.dot(K12)/sigma2
    logdeterminantS = np.linalg.slogdet(A)[1] - np.linalg.slogdet(K22)[1] + N*np.log(sigma2)
    B = sigma2**(-1)* np.eye(N) - sigma2**(-2)*(K12.dot(np.linalg.inv(A)).dot(K21)) 
    res = -0.5*N*np.log(2*np.pi) - 0.5*logdeterminantS - 0.5*Y.T.dot(B.dot(Y))
    return res[0][0]


def log_Sparse_Guassian_validation(Y, K12, K22, K21, sigma2):
    N, M = K12.shape
    Q = K12.dot(np.linalg.inv(K22)).dot(K21) 
    return multivariate_normal.logpdf(Y.reshape(-1), mean=np.zeros_like(Y.reshape(-1)), cov=Q+sigma2*np.eye(N))


def log_likelihood(hyper_parameters, X, Y, Z, kernel=RBF_cov):
    log_sigma2_err, log_sigma2, log_l = hyper_parameters
    sigma2_err = np.exp(log_sigma2_err)
    sigma2 = np.exp(log_sigma2)
    l = np.exp(log_l)
    K12 = kernel(X, Z, sigma2, l)
    # import pdb
    # pdb.set_trace()
    K21 = K12.T
    K22 = kernel(Z, Z, sigma2, l)
    llk = log_Sparse_Guassian(Y, K12, K22, K21, sigma2_err)
    # vllk = log_Sparse_Guassian_validation(Y, K12, K22, K21, sigma2)
    # print("llk: {} and validated llk: {}".format(llk, vllk))
    Q = K12.dot(np.linalg.inv(K22)).dot(K21)
    return llk - 0.5/sigma2*np.sum(sigma2 - np.diag(Q))

    
def objective_function_hyper(pars, X, Y, Z, kernel=RBF_cov):
    return -log_likelihood(pars, X, Y, Z, kernel)


def objective_function_hyperandIPs(pars, X, Y, kernel=RBF_cov):
    hyper_parameters = pars[:3]
    Z = pars[3:]
    return -log_likelihood(hyper_parameters, X, Y, Z, kernel)


def regularization(X, Z):
    if len(X.shape) == 1:
        X = X.reshape([-1, 1])
    if len(Z.shape) == 1:
        Z = Z.reshape([-1, 1])    
    dist = distance.cdist(X, Z)
    res = np.min(dist, axis = 1)
    return np.sum(res)


def objective_function_hyperandIPs_reg(pars, X, Y, lamb, kernel=RBF_cov):
    hyper_parameters = pars[:3]
    Z = pars[3:]
    return -log_likelihood(hyper_parameters, X, Y, Z, kernel) + lamb*regularization(X, Z)


def training_hyper(X, Y, Z, hyper_parameters=np.zeros(3), kernel=RBF_cov):
    # initialize parameters
    pars = hyper_parameters
    print(objective_function_hyper(hyper_parameters, X, Y, Z, kernel))
    res = minimize(objective_function_hyper, pars, args = (X, Y, Z, kernel), method='BFGS', options={'xtol': 1e-8, 'disp': True})
    return res.x


def training_hyperandIPs(X, Y, Z, hyper_parameters=-np.zeros(3), kernel=RBF_cov):
    # intialize parameters
    pars = np.concatenate([hyper_parameters, Z])
    print(objective_function_hyperandIPs(hyper_parameters, X, Y, kernel))
    res = minimize(objective_function_hyperandIPs, pars, args = (X, Y, kernel), method='BFGS', options={'xtol': 1e-8, 'disp': True})
    return res.x


def training_hyperandIPs_reg(X, Y, Z, hyper_parameters=np.zeros(3), lamb=1, kernel=RBF_cov):
    # intialize parameters
    pars = np.concatenate([hyper_parameters, Z])
    print(objective_function_hyperandIPs_reg(pars, X, Y, lamb, kernel))
    res = minimize(objective_function_hyperandIPs_reg, pars, args = (X, Y, lamb, kernel), method='BFGS', options={'xtol': 1e-8, 'disp': True})
    return res.x    


def prediction(X, Y, Z, X_pred, hyper_parameters, kernel=RBF_cov):
    N = Y.shape[0]
    log_sigma2_err, log_sigma2, log_l = hyper_parameters
    sigma2_err = np.exp(log_sigma2_err)
    sigma2 = np.exp(log_sigma2)
    l = np.exp(log_l)
    K12 = kernel(X, Z, sigma2, l)
    K21 = K12.T
    K22 = kernel(Z, Z, sigma2, l)
    A = K22 + K21.dot(K12)/sigma2_err
    y = Y.reshape(-1)
    mu = sigma2_err**(-1)*K22.dot(np.linalg.solve(A, K21.dot(y))) 
    mu_pred = kernel(X_pred, Z, sigma2, l).dot(np.linalg.solve(K22, mu))
    return mu_pred


from sklearn.model_selection import KFold
def cross_validation(X, Y, init_Z=np.linspace(0, 1, 10), init_hyper_parameters=np.array([-2,-2,-2]), lamb=1., kernel=RBF_cov, n_fold = 5):
    kf = KFold(n_splits=n_fold)
    rmses = list()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]    
        pars = training_hyperandIPs_reg(X_train, y_train, init_Z, init_hyper_parameters, lamb, kernel=kernel)    
        hyper_parameters = pars[:3]
        Z = pars[3:]
        Preds = prediction(X, Y, Z, X_test, hyper_parameters, kernel=kernel)
        rmse = np.sqrt(np.mean((y_test - Preds)**2))
        rmses.append(rmse)
    return np.mean(np.array(rmses))


if __name__ == "__main__":
    with open("SIM_1D.pickle", "rb") as res:
        [X, Y, X_validation, Y_vailidation, Xs, Fs] = pickle.load(res)
    # with open("SIM_1D_clustered.pickle", "rb") as res:
    #     [X, Y, X_validation, Y_vailidation, Xs, Fs] = pickle.load(res)
    kernel = Matern32_cov

    Fs = Fs.reshape(-1)
    n_IPs = 10
    init_Z = np.linspace(0, 1, n_IPs)
    init_hyper_parameters = np.array([-2, -2, -2])


    ### Train hyper parameters
    pars = hyper_parameters = training_hyper(X, Y, init_Z, init_hyper_parameters, kernel=kernel)    
    Preds = prediction(X, Y, init_Z, Xs, hyper_parameters, kernel=kernel)
    print("1, RMSE: {}".format(np.sqrt(np.mean((Fs - Preds)**2))))
    # figure = plt.figure()
    # plt.scatter(X, Y)
    # plt.plot(Xs, Fs, color='r')
    # plt.plot(Xs, Preds, color='b')
    # plt.scatter(Z, np.zeros_like(Z), marker="x")
    # plt.show()

    ### Train hyper parameters and inducing inputs
    pars = training_hyperandIPs(X, Y, init_Z, init_hyper_parameters, kernel=kernel)    
    hyper_parameters = pars[:3]
    Z = pars[3:]
    print("similarity", regularization(X, Z))
    Preds = prediction(X, Y, Z, Xs, hyper_parameters, kernel=kernel)
    print("2, RMSE: {}".format(np.sqrt(np.mean((Fs - Preds)**2))))
    # figure = plt.figure()
    # plt.scatter(X, Y)
    # plt.plot(Xs, Fs, color='r')
    # plt.plot(Xs, Preds, color='b')
    # plt.scatter(Z, np.zeros_like(Z), marker="x")
    # plt.show()

    # ### Select optimal lambda using validation dataset
    # lamb_set = np.linspace(0,5,10)
    # score = list()
    # for lamb in lamb_set:
    #     Z = np.linspace(0, 1, n_IPs)
    #     hyper_parameters = np.array([-2, -2, -2])
    #     pars = training_hyperandIPs_reg(X, Y, Z, hyper_parameters, lamb, kernel=kernel)    
    #     hyper_parameters = pars[:3]
    #     Z = pars[3:]
    #     Preds = prediction(X, Y, Z, X_validation, hyper_parameters, kernel=kernel)
    #     rmse = np.sqrt(np.mean((Y_vailidation - Preds)**2))
    #     score.append(rmse)
    # score = np.array(score)
    # opt_lamb = lamb_set[np.argmin(score)]
    # print(score)
    # print("best lambda: {}".format(opt_lamb))

    # Select optimal lambda using cross validation
    lamb_set = np.arange(10)
    rmses = list()
    for lamb in lamb_set:
        rmse = cross_validation(X, Y, init_Z, init_hyper_parameters, lamb = lamb)
        rmses.append(rmse)
    opt_lamb = lamb_set[np.argmin(rmses)]
    print("best lambda: {}".format(opt_lamb))

    # Prediction
    pars = training_hyperandIPs_reg(X, Y, init_Z, init_hyper_parameters, lamb=opt_lamb, kernel=kernel)    
    hyper_parameters = pars[:3]
    Z = pars[3:]
    print("similarity", regularization(X, Z))
    Preds = prediction(X, Y, Z, Xs, hyper_parameters, kernel=kernel)
    print("3, RMSE: {}".format(np.sqrt(np.mean((Fs - Preds)**2))))
    figure = plt.figure()
    plt.scatter(X, Y)
    plt.plot(Xs, Fs, color='r')
    plt.plot(Xs, Preds, color='b')
    plt.scatter(Z, np.zeros_like(Z), marker="x")
    plt.show()
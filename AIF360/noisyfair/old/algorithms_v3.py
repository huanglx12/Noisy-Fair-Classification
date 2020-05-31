import sys
sys.path.append("../")
from scipy.optimize import minimize # for loss func minimization
import numpy as np
import aif360.algorithms.inprocessing.zvrg.zvrg_utils as zvrg_ut
import aif360.algorithms.inprocessing.zvrg.zvrg_loss_funcs as zvrg_lf
import aif360.algorithms.inprocessing.gyf.gyf_utils as gyf_ut
import aif360.algorithms.inprocessing.gyf.gyf_loss_funcs as gyf_lf

import random
random.seed()
np.seterr(divide = 'ignore', invalid='ignore')

####################################################
# tools
####################################################
# Sigmoid function
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

# log loss
def log_logistic(X):

	""" This function is used from scikit-learn source code. Source link below """

	"""Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
	This implementation is numerically stable because it splits positive and
	negative values::
	    -log(1 + exp(-x_i))     if x_i > 0
	    x_i - log(1 + exp(x_i)) if x_i <= 0

	Parameters
	----------
	X: array-like, shape (M, N)
	    Argument to the logistic function

	Returns
	-------
	out: array, shape (M, N)
	    Log of the logistic function evaluated at every point in x
	Notes
	-----
	Source code at:
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
	-----

	See the blog post describing this implementation:
	http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
	"""
	if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
	out = np.empty_like(X) # same dimensions and data types

	idx = X>0
	out[idx] = -np.log(1.0 + np.exp(-X[idx]))
	out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
	return out

####################################################
# algorithms
####################################################
# ZVRG
def zvrg(features, labels, index, C, thresh):
    N = features.shape[0]
    d = features.shape[1]
    # X = np.asarray(features)
    X = np.zeros([N, d+1])
    X[:,0:d] = features
    X[:,d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # loss_function = lf._rosen
    loss_function = zvrg_lf._rosen
    x_control_train = {"attr": features[:,index]}
    mode = {"fairness": 1, "lam": float(C), 'is_reg': 1}
    sensitive_attrs = list(x_control_train.keys())
    sensitive_attrs_to_cov_thresh = dict((k, thresh) for (k, v) in x_control_train.items())
    theta = zvrg_ut.train_model(X, labels, x_control_train, loss_function,
                           mode.get('fairness', 0),
                           mode.get('accuracy', 0),
                           mode.get('separation', 0),
                           sensitive_attrs,
                           sensitive_attrs_to_cov_thresh,
                           mode.get('gamma', None),
                           mode.get('lam', None),
                           mode.get('is_reg', 0))
    return theta


#####################################################
def gyf(features, labels, index, C, thresh):
    N = features.shape[0]
    d = features.shape[1]
    # X = np.asarray(features)
    X = np.zeros([N, d + 1])
    X[:, 0:d] = features
    X[:, d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # loss_function = lf._logistic_loss_l2_reg
    loss_function = gyf_lf._fair_logistic_loss_l2
    x_control_train = {"attr": features[:, index]}
    mode = {"fairness": 2, "lam": float(C), 'is_reg': 1, 'gamma': float(thresh)}
    sensitive_attrs = list(x_control_train.keys())
    sensitive_attrs_to_cov_thresh = {}
    theta = gyf_ut.train_model(X, labels, x_control_train, loss_function,
                                mode.get('fairness', 0),
                                mode.get('accuracy', 0),
                                mode.get('separation', 0),
                                sensitive_attrs,
                                sensitive_attrs_to_cov_thresh,
                                mode.get('gamma', 1),
                                mode.get('lam', None),
                                mode.get('is_reg', 0))
    return theta


#####################################################
# solving denoised fair program based on GYF
def denoised_gyf(features, labels, index, C, thresh, eta0, eta1):
    N = features.shape[0]
    d = features.shape[1]
    x_control = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        if int(features[i][index]) == 1:
            x_control[i] = 1
        else:
            x_control[i] = 0
        if int(labels[i]) == 1:
            y[i] = 1
        else:
            y[i] = 0
    X = np.zeros([N, d+1])
    X[:,0:d] = features
    X[:,d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # compute fair_reg0 and fair_reg1
    reg_hat0 = float(len(set(list(np.where(x_control == 0)[0])).intersection(
        set(list(np.where(y == 0)[0])))))
    reg_hat1 = float(len(set(list(np.where(x_control == 1)[0])).intersection(
        set(list(np.where(y == 0)[0])))))
    reg0 = float((1 - eta1) * reg_hat0 - eta1 * reg_hat1) / (1 - eta0 - eta1)
    reg1 = float((1 - eta0) * reg_hat1 - eta0 * reg_hat0) / (1 - eta0 - eta1)
    # print(reg0, reg1)
    mu_hat1 = np.sum(x_control)
    mu_hat0 = N - mu_hat1
    mu0 = float((1 - eta1) * mu_hat0 - eta1 * mu_hat1) / (1 - eta0 - eta1)
    mu1 = float((1 - eta0) * mu_hat1 - eta0 * mu_hat0) / (1 - eta0 - eta1)
    # print(mu0, mu1)
    fair_reg0 = reg0 / mu0 / mu0
    fair_reg1 = reg1 / mu1 / mu1

    p00 = (1 - eta0) * mu0 / mu_hat0
    p01 = eta1 * mu1 / mu_hat0
    p10 = eta0 * mu0 / mu_hat1
    p11 = (1 - eta1) * mu1 / mu_hat1
    # coeff
    coeff0 = p00 * fair_reg0 + p01 * fair_reg1
    coeff1 = p10 * fair_reg0 + p11 * fair_reg1
    print(coeff0 * N, coeff1 * N)

    # loss function
    def rosen(x):
        obj = 0
        for i in range(N):
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            obj -= (label * np.log(sigma) + (1-label) * np.log(1-sigma)) / N
            if x_control[i] == 0:
                obj -= coeff0 * np.log(sigma)
            else:
                obj -= coeff1 * np.log(sigma)
        for i in range(d):
            obj += C * x[i]**2
        return obj

    x0 = np.random.rand(d)
    # res = minimize(logistic, x0, method='SLSQP', jac=logistic_der, constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
    res = minimize(fun = rosen, x0 = x0, method='SLSQP', constraints = [], options = {'maxiter': 200, 'ftol': 1e-9, 'disp': True})
    return res.x

########################################################
# solving denoised fair program based on ZVRG
def denoised_zvrg(features, labels, index, C, thresh, lam, eta0, eta1):
    N = features.shape[0]
    d = features.shape[1]
    x_control = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        if int(features[i][index]) == 1:
            x_control[i] = 1
        else:
            x_control[i] = 0
        if int(labels[i]) == 1:
            y[i] = 1
        else:
            y[i] = 0
    X = np.zeros([N, d+1])
    X[:,0:d] = features
    X[:,d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)
    per = (1 - eta0 - eta1) * lam - 0.01

    mu_hat1 = float(np.sum(x_control)) / N
    mu_hat0 = 1 - mu_hat1
    mu0 = float((1 - eta1) * mu_hat0 - eta1 * mu_hat1) / (1 - eta0 - eta1)
    mu1 = float((1 - eta0) * mu_hat1 - eta0 * mu_hat0) / (1 - eta0 - eta1)
    # print(mu0, mu1)
    p00 = (1 - eta0) * mu0 / mu_hat0
    p01 = eta1 * mu1 / mu_hat0
    p10 = eta0 * mu0 / mu_hat1
    p11 = (1 - eta1) * mu1 / mu_hat1
    coeff0 = - p00 * mu1 + p01 * (1 - mu1)
    coeff1 = - p10 * mu1 + p11 * (1 - mu1)

    # loss function
    def rosen(x):
        obj = 0
        for i in range(N):
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            obj -= (label * np.log(sigma) + (1 - label) * np.log(1 - sigma)) / N
        for i in range(d):
            obj += C * x[i] ** 2
        return obj

    # denoised fairness constraints
    def cons_f(x):
        f = np.array([thresh, thresh, -per, -per])
        Sigma = np.array([sigmoid(np.dot(x, X[i])) for i in range(N)])
        for i in range(N):
            fea = X[i]
            if x_control[i] == 0:
                f[0] -= coeff0 * np.dot(x, fea)
                f[1] += coeff0 * np.dot(x,fea)
                f[2] += (1-eta1) * Sigma[i] / N
                f[3] += - eta1 * Sigma[i] / N
            else:
                f[0] -= coeff1 * np.dot(x, fea)
                f[1] += coeff1 * np.dot(x,fea)
                f[3] += - eta0 * Sigma[i] / N
                f[4] += (1 - eta0) * Sigma[i] / N
        return f

    x0 = np.random.rand(d)
    ineq_cons = {'type': 'ineq', 'fun': lambda x: cons_f(x)}
    res = minimize(fun=rosen, x0=x0, method='SLSQP', constraints=[ineq_cons],
                   options={'maxiter': 200, 'ftol': 1e-9, 'disp': True})
    return res.x

# test
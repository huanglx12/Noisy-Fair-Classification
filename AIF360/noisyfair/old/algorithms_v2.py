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
# solving denoised fair program
def denoised(features, labels, index, C, tau, lam, eta0, eta1):
    N = features.shape[0]
    d = features.shape[1]
    N1 = sum(features[:,index])
    N0 = N - N1
    mu0 = (1 - eta1) * N0 / N - eta1 * N1 / N
    mu1 = (1 - eta0) * N1 / N - eta0 * N0 / N
    coeff00 = (tau - 0.01) * eta0 * mu0 + (1 - eta1) * mu1
    coeff01 = - (tau - 0.01) * (1 - eta0) * mu0 - eta1 * mu1
    coeff10 = - (tau - 0.01) * (1 - eta1) * mu1 - eta0 * mu0
    coeff11 = (tau - 0.01) * eta1 * mu1 + (1 - eta0) * mu0
    per = (1 - eta0 - eta1) * lam - 0.01
    X = np.zeros([N, d+1])
    X[:,0:d] = features
    X[:,d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # # loss function
    # def logistic_loss_l2_reg(w):
    #     yz = np.zeros(N)
    #     for i in range(N):
    #         yz[i] = (2 * labels[i] - 1) * np.dot(X[i], w)
    #     # Logistic loss is the negative of the log of the logistic function.
    #     logistic_loss = -np.sum(log_logistic(yz))
    #     l2_reg = (float(C) * N) * np.sum([elem * elem for elem in w])
    #     out = logistic_loss + l2_reg
    #     return out

    def rosen(x):
        obj = 0
        for i in range(N):
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            obj -= label * np.log(sigma) + (1-label) * np.log(1-sigma)
        obj /= N
        for i in range(d):
            obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        for i in range(N):
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            der += (sigma-label) * fea
        der = der / N
        for i in range(d):
            der[i] += 2 * C * x[i]
        return der

    # denoised fairness constraints
    def cons_f(x):
        f = np.zeros(4)
        Sigma = np.array([sigmoid(np.dot(x, X[i])) for i in range(N)])
        for i in range(N):
            if int(features[i][index]) == 0:
                f[0] += (1-eta1) * Sigma[i] / N
                f[1] += - eta0 * Sigma[i] / N
                f[2] += coeff00 * Sigma[i]
                f[3] += coeff10 * Sigma[i]
            else:
                f[0] += - eta1 * Sigma[i] / N
                f[1] += (1 - eta0) * Sigma[i] / N
                f[2] += coeff01 * Sigma[i]
                f[3] += coeff11 * Sigma[i]
        f[0] -= per
        f[1] -= per
        return f

    def cons_J(x):
        J = np.array([[0.0 for i in range(d)] for j in range(4)])
        temp = np.array([sigmoid(np.dot(x, X[i])) for i in range(N)])
        Sigma = np.array([temp[i] * (1 - temp[i]) for i in range(N)])
        for i in range(N):
            fea = X[i]
            if int(features[i][index]) == 0:
                J[0] += ((1-eta1) * Sigma[i] / N) * fea
                J[1] += - (eta0 * Sigma[i] / N) * fea
                J[2] += coeff00 * Sigma[i] * fea
                J[3] += coeff10 * Sigma[i] * fea
            else:
                J[0] += - (eta1 * Sigma[i] / N) * fea
                J[1] += ((1 - eta0) * Sigma[i] / N) * fea
                J[2] += coeff01 * Sigma[i] * fea
                J[3] += coeff11 * Sigma[i] * fea
        return J

    x0 = np.random.rand(d)
    ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_f(x), 'jac' : lambda x: cons_J(x)}
    # res = minimize(logistic, x0, method='SLSQP', jac=logistic_der, constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
    res = minimize(fun = rosen, x0 = x0, method='SLSQP', jac = rosen_der, constraints = [ineq_cons], options = {'maxiter': 200, 'ftol': 1e-9, 'disp': True})
    return res.x


###############################################################
def undenoised(features, labels, index, C, tau):
    N = features.shape[0]
    d = features.shape[1]
    N1 = sum(features[:, index])
    N0 = N - N1
    X = np.zeros([N, d + 1])
    X[:, 0:d] = features
    X[:, d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # # logistic loss
    # def logistic_loss_l2_reg(w):
    #     yz = np.zeros(N)
    #     for i in range(N):
    #         yz[i] = (2 * labels[i] - 1) * np.dot(X[i], w)
    #     # Logistic loss is the negative of the log of the logistic function.
    #     logistic_loss = -np.sum(log_logistic(yz))
    #     l2_reg = (float(C) * N) * np.sum([elem * elem for elem in w])
    #     out = logistic_loss + l2_reg
    #     return out

    # loss function
    def rosen(x):
        obj = 0
        for i in range(N):
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            obj -= label * np.log(sigma) + (1-label) * np.log(1-sigma)
        obj /= N
        for i in range(d):
            obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        for i in range(N):
            fea = X[i]
            label = labels[i]
            sigma = sigmoid(np.dot(x, fea))
            der += (sigma-label) * fea
        der = der / N
        for i in range(d):
            der[i] += 2 * C * x[i]
        return der

    # fairness constraints
    def cons_f(x):
        f = np.zeros(2)
        Sigma = np.array([sigmoid(np.dot(x, X[i])) for i in range(N)])
        for i in range(N):
            if int(features[i][index]) == 0:
                f[0] += Sigma[i] / N0
                f[1] += - tau * Sigma[i] / N0
            else:
                f[0] += - tau * Sigma[i] / N1
                f[1] += Sigma[i] / N1
        return f

    def cons_J(x):
        J = np.array([[0.0 for i in range(d)] for j in range(2)])
        temp = np.array([sigmoid(np.dot(x, X[i])) for i in range(N)])
        Sigma = np.array([temp[i] * (1 - temp[i]) for i in range(N)])
        for i in range(N):
            fea = X[i]
            if int(features[i][index]) == 0:
                J[0] += Sigma[i] * fea / N
                J[1] += - tau * Sigma[i] * fea / N
            else:
                J[0] += - tau * Sigma[i] * fea / N
                J[1] += Sigma[i] * fea / N
        return J

    x0 = np.random.rand(d)
    ineq_cons = {'type': 'ineq', 'fun': lambda x: cons_f(x), 'jac': lambda x: cons_J(x)}
    res = minimize(fun=rosen, x0=x0, method='SLSQP', jac = rosen_der, constraints=[ineq_cons],
                 options={'maxiter': 200, 'ftol': 1e-9, 'disp': True})
    return res.x


# test
import sys
sys.path.append("../")
from scipy.optimize import NonlinearConstraint, minimize
import numpy as np
import aif360.algorithms.inprocessing.zvrg.zvrg_utils as zvrg_ut
import aif360.algorithms.inprocessing.zvrg.zvrg_loss_funcs as zvrg_lf
import aif360.algorithms.inprocessing.gyf.gyf_utils as gyf_ut
import aif360.algorithms.inprocessing.gyf.gyf_loss_funcs as gyf_lf

import random
random.seed()
np.seterr(divide = 'ignore', invalid='ignore')

####################################################
# Sigmoid function
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

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

    # loss_function = lf._logistic_loss_l2_reg
    loss_function = zvrg_lf._logistic_loss_l2_reg
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
    mu0 = (1 - eta1) * N0 / N - eta0 * N1 / N
    mu1 = (1 - eta0) * N1 / N - eta1 * N0 / N
    coeff00 = (tau - 0.01) * eta1 * mu0 + (1 - eta1) * mu1
    coeff01 = - (tau - 0.01) * (1 - eta0) * mu0 - eta0 * mu1
    coeff10 = - (tau - 0.01) * (1 - eta1) * mu1 - eta1 * mu0
    coeff11 = (tau - 0.01) * eta0 * mu1 + (1 - eta0) * mu0
    per = (1 - eta0 - eta1) * lam - 0.01
    X = np.zeros([N, d+1])
    X[:,0:d] = features
    X[:,d] = [1.0 for i in range(N)]
    X = np.delete(X, index, 1)

    # def logistic(x):
    #     obj = 0
    #     for i in range(N):
    #         fea = X[i]
    #         label = labels[i]
    #         sigma = 1 / (1 + np.exp(-label * np.dot(x, fea)))
    #         obj -= np.log(sigma)
    #     obj = obj / N
    #     for i in range(d):
    #         obj += C * x[i]**2
    #     return obj
    #
    # def logistic_der(x):
    #     der = np.zeros(d)
    #     for i in range(N):
    #         fea = X[i]
    #         label = labels[i]
    #         sigma = np.exp(- label * np.dot(x, fea)) / (1 + np.exp(- label * np.dot(x, fea)))
    #         der -= label * sigma * fea
    #     der = der / N
    #     for i in range(d):
    #         der[i] += 2 * C * x[i]
    #     return der

    # logistic loss
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
        der = der/N
        for i in range(d):
            der[i] += 2 * C * x[i]
        return der

    # def rosen_hess(x):
    #     D = np.zeros(N)
    #     H = np.zeros([d,d])
    #     for i in range(N):
    #         fea = X[i]
    #         sigma = 1 / (1 + np.exp(-np.dot(x, fea)))
    #         D[i] = sigma * (1 - sigma)
    #     for r in range(d):
    #         for c in range(d):
    #             for i in range(N):
    #                 H[r][c] += X[i][r] * X[i][c] * D[i]
    #     H = H / N
    #     for i in range(d):
    #         H[i][i] += 2 * C
    #     return H

    # denoised fairness constraints
    def cons_f(x):
        f = np.zeros(4)
        Sigma = np.array([sigmoid(np.dot(x, X[i])) for i in range(N)])
        for i in range(N):
            if int(features[i][index]) == 0:
                f[0] += (1-eta1) * Sigma[i] / N
                f[1] += - eta1 * Sigma[i] / N
                f[2] += coeff00 * Sigma[i]
                f[3] += coeff10 * Sigma[i]
            else:
                f[0] += - eta0 * Sigma[i] / N
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
                J[1] += - (eta1 * Sigma[i] / N) * fea
                J[2] += coeff00 * Sigma[i] * fea
                J[3] += coeff10 * Sigma[i] * fea
            else:
                J[0] += - (eta0 * Sigma[i] / N) * fea
                J[1] += ((1 - eta0) * Sigma[i] / N) * fea
                J[2] += coeff01 * Sigma[i] * fea
                J[3] += coeff11 * Sigma[i] * fea
        return J

    # def cons_H(x, v):
    #     H = np.zeros([d,d])
    #     temp = np.array([1 / (1 + np.exp(-np.dot(x, X[i]))) for i in range(N)])
    #     Sigma = np.array([temp[i] * (1 - temp[i]) * (1 - 2 * temp[i]) for i in range(N)])
    #     Sigma0 = np.array([0.0 for i in range(N)])
    #     Sigma1 = np.array([0.0 for i in range(N)])
    #     Sigma2 = np.array([0.0 for i in range(N)])
    #     Sigma3 = np.array([0.0 for i in range(N)])
    #     for i in range(N):
    #         if int(features[i][index]) == 0:
    #             Sigma0[i] = (1 - eta1) * Sigma[i] / N
    #             Sigma1[i] = - eta1 * Sigma[i] / N
    #             Sigma2[i] = coeff00 * Sigma[i]
    #             Sigma3[i] = coeff10 * Sigma[i]
    #         else:
    #             Sigma0[i] = - eta0 * Sigma[i] / N
    #             Sigma1[i] = (1 - eta0) * Sigma[i] / N
    #             Sigma2[i] = coeff01 * Sigma[i]
    #             Sigma3[i] = coeff11 * Sigma[i]
    #     for r in range(d):
    #         for c in range(d):
    #             for i in range(N):
    #                 H[r][c] += v[0] * X[i][r] * X[i][c] * Sigma0[i]
    #                 H[r][c] += v[1] * X[i][r] * X[i][c] * Sigma1[i]
    #                 H[r][c] += v[2] * X[i][r] * X[i][c] * Sigma2[i]
    #                 H[r][c] += v[3] * X[i][r] * X[i][c] * Sigma3[i]
    #     return H

    x0 = np.zeros(d)
    # lower = np.array([0, 0, 0, 0])
    # upper = np.array([np.inf, np.inf, np.inf, np.inf])
    # nonlinear_constraint = NonlinearConstraint(cons_f, lower, upper, jac=cons_J, hess=cons_H)
    # res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
    #                constraints = [nonlinear_constraint], options = {'gtol': 1e-1, 'xtol': 1e-1, 'verbose': 3})
    ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_f(x), 'jac' : lambda x: cons_J(x)}
    # res = minimize(logistic, x0, method='SLSQP', jac=logistic_der, constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
    res = minimize(rosen, x0, method='SLSQP', jac=rosen_der, constraints = [ineq_cons], options = {'maxiter': 200, 'ftol': 1e-9, 'disp': True})
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

    # logistic loss
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
        der = der/N
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

    x0 = np.zeros(d)
    ineq_cons = {'type': 'ineq', 'fun': lambda x: cons_f(x), 'jac': lambda x: cons_J(x)}
    res = minimize(rosen, x0, method='SLSQP', jac=rosen_der, constraints=[ineq_cons], options={'maxiter': 200, 'ftol': 1e-9, 'disp': True})
    return res.x


# test
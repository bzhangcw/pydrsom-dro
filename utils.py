import logging

import torch
import time
import glob
import numpy as np
import torch.nn.functional as F
import cvxpy as cp

# DEFAULT_SOLVER=cp.GUROBI
DEFAULT_SOLVER = cp.OSQP


def save_model(model, args, path):
    # torch.save(model.state_dict(),args.save_path+path)
    # if is_best:
    torch.save(model.state_dict(), args.save_path + "/" + path)


def chi_square_func(lbda, loss, eta):
    obj = (loss - eta) / lbda + 2
    obj[obj < 0] = 0
    obj = lbda * (-1 + 1 / 4 * torch.pow(obj, 2)) + eta
    return obj


def chi_square_grad(lbda, loss, eta):
    grad = (loss - eta) / lbda + 2
    grad[grad < 0] = 0
    grad = 1 - grad / 2
    return grad


def get_eta(loss_vec, lbda, init_eta=0.002, lr=0.08):
    eta = init_eta
    iter = 0
    grad_func = chi_square_grad
    grad = grad_func(lbda, loss_vec, eta).mean().item()
    while abs(grad) > 1e-7 and iter < 1000:
        eta = eta - lr * grad
        grad = grad_func(lbda, loss_vec, eta).mean().item()
        iter += 1

    # assert abs(eta - get_eta_by_cvxpy(loss_vec, lbda)) < 1e-3
    return np.array(eta)


def get_eta_by_cvxpy(loss_vec, lbda):
    n = loss_vec.shape[0]
    ee = np.ones(n)
    x = cp.Variable(n)
    eta = cp.Variable()
    l = loss_vec
    constraints = [x >= 0, x - ee * 2 >= l / lbda - eta * ee / lbda]
    objective = cp.Minimize(lbda * cp.sum_squares(x) / 4 / n + eta)
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    _ = prob.solve(solver=DEFAULT_SOLVER)
    # The optimal value for x is stored in `x.value`.
    return eta.value


def DRO_cross_entropy(predict, label, lbda):
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    entropy_vec = loss_func(predict, label)
    try:
        eta = get_eta_by_cvxpy(entropy_vec.cpu().detach().numpy(), lbda)
    except Exception as e:
        logging.exception(e)
        print("error in cvxpy")
        eta = get_eta(entropy_vec.cpu().detach().numpy(), lbda, 0, 0.05)
    loss_vec = chi_square_func(lbda, entropy_vec, torch.from_numpy(eta))
    loss = loss_vec.mean()
    return loss

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


def imbalance_sampling(dataset):
    # ratios = {k: np.random.rand() * 0.8 + 0.2 for k in dataset.targets.unique().numpy()}
    imbalance_ratio = [
        0.738,
        0.986,
        0.446,
        0.254,
        0.768,
        0.593,
        0.918,
        0.731,
        0.929,
        0.284,
    ]
    try:
        ratios = dict(zip(sorted(dataset.targets.unique().numpy().tolist()), imbalance_ratio))
        tt = dataset.targets.numpy()
    except:
        ratios = dict(zip(sorted(np.unique(dataset.targets).tolist()), imbalance_ratio))
        tt = np.array(dataset.targets)
    # sample by the desired ratio
    bool_select = [np.random.rand() > 1 - ratios[u] for u in tt]
    y = tt[bool_select]
    x = dataset.data[bool_select]
    print(
        f"""sampling statistics:
    original: {np.histogram(dataset.targets)[0]}
    ρ       : {np.array(list(ratios.values())).round(3)}
    after   : {np.histogram(y)[0]}
    total   : {y.shape[0]}
    """
    )
    dataset.data = x
    dataset.targets = y
    return dataset, imbalance_ratio


######################
# χ2 optimization
######################
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


def get_eta(loss_vec, lbda, grad_func, init_eta=0.002, lr=0.01):
    eta = init_eta
    iter = 0
    grad = grad_func(lbda, loss_vec, eta).mean().item()
    while abs(grad) > 1e-5 and iter < 1000:
        eta = eta - lr * grad
        grad = grad_func(lbda, loss_vec, eta).mean().item()
        iter += 1
    return eta


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
        grad_func = chi_square_grad
        obj_func = chi_square_func
        eta = get_eta(entropy_vec.cpu().detach().numpy(), lbda, grad_func, 0, 0.05)
    loss_vec = chi_square_func(lbda, entropy_vec, torch.from_numpy(eta))
    loss = loss_vec.mean()
    return loss


######################
# CVaR optimization
######################
def CVaR_cross_entropy(predict, label, lbda=1.0):
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    entropy_vec = loss_func(predict, label)
    grad_func = smoothed_CVaR_grad
    obj_func = smoothed_CVaR_func
    vv = entropy_vec.cpu().detach().numpy()
    eta = get_eta(vv, lbda, grad_func, 0, 0.05)
    loss_vec = obj_func(lbda, entropy_vec, torch.from_numpy(np.array(eta)))
    loss = loss_vec.mean()
    return loss


def smoothed_CVaR_func(lbda, loss, eta, alpha=0.05):
    obj = (loss - eta) / lbda
    obj = torch.log(1 - alpha + alpha * torch.exp(obj)) / alpha
    obj = lbda * obj + eta
    return obj


def smoothed_CVaR_grad(lbda, loss, eta, alpha=0.05):
    grad = (loss - eta) / lbda
    grad = 1 - np.exp(grad) / (1 - alpha + alpha * np.exp(grad))
    return grad

import time
import json
from functools import reduce
from pprint import pprint

import torch
import csv
from torch.utils.data import DataLoader
import argparse

import utils
from my_model import CNNModel, LogisticRegression
from utils import *
from torchvision import datasets, transforms
from pydrsom.drsom import DRSOMB as DRSOM
from pydrsom.drsom_utils import *
import matplotlib.pyplot as plt
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_args():
    parser = argparse.ArgumentParser(
        description="argument parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        choices=["MNIST", "FMNIST"],
        help="select dataset",
        default="FMNIST",
    )
    parser.add_argument(
        "--imbalance",
        required=False,
        type=int,
        help="whether to sample an imbalanced data, default true ",
        default=1,
    )
    parser.add_argument(
        "--model",
        required=False,
        type=str,
        choices=["cnn", "log"],
        help="select loss function",
        default="cnn",
    )
    parser.add_argument(
        "--lossfunc",
        required=False,
        type=str,
        choices=["dro", "dro-cvar", "usual"],
        help="select loss function",
        default="dro",
    )
    parser.add_argument(
        "--save_path",
        default=r"/tmp/",
    )
    parser.add_argument(
        "--optim",
        required=False,
        type=str,
        default="nsgd",
        choices=["adam", "sgd", "drsom", "nsgd", "sgd_clip"],
    )
    parser.add_argument(
        "--mu", default=1e-3, type=float, help="μ, only use for clipped SGD"
    )
    parser.add_argument(
        "--vr_ratio",
        required=False,
        type=int,
        default=-1,
        help="""
      the ratio of variance reduction,
        if ever VR is enabled.
      i.e., if let γ be the ratio (if > 0)
        then in probability 1/γ we will use the vr iteration.
      """,
    )
    parser.add_argument(
        "--vr_batch_ratio",
        required=False,
        type=int,
        default=20,
        help="""
          the batch size ratio of variance reduction relative to normal batch size,
            if ever VR is enabled.
          i.e., if let γ be the ratio,
            then in probability γ we will use the vr iteration.
          """,
    )
    parser.add_argument("--epoch", default=25, type=int, help="epoch number")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--trained_model", default=None, help="the path to the saved trained model"
    )
    parser.add_argument(
        "--interval", required=False, type=int, default=50, help="logging interval"
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument(
        "--gamma", default=1e-3, type=float, help="hyper param for clipped method"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--run_id", default=1, type=int, help="repetition id")
    add_parser_options(parser)
    args = parser.parse_args()
    return args


def train(dataloader, name, model, loss_fn, optimizer, args, train_large_loader=None):
    model.train()
    if name.startswith("drsom"):
        return train_drsom(dataloader, model, loss_fn, optimizer, args.interval)
    st = time.time()
    size = len(dataloader.dataset)

    correct = 0
    total = 0
    avg_loss = 0
    iter_large = iter(train_large_loader) if train_large_loader is not None else None
    vr_batch_numbers = []
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        bool_switch = False
        if (iter_large is not None) and ((batch + 1) % args.vr_ratio == 0):
            # override with large batch
            X, y = next(iter_large)
            bool_switch = True
            vr_batch_numbers.append(batch)
        X, y = X.to(device), y.to(device)

        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            return loss

        # backpropagation
        if not bool_switch:
            loss = optimizer.step(closure=closure)
        else:
            loss = optimizer.step(bool_switch=bool_switch, closure=closure)
        avg_loss += loss.item()
        losses.append(loss.item())
        # compute prediction error
        outputs = model(X)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        if batch % args.interval == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    accuracy = 100.0 * correct / total
    avg_loss = avg_loss / len(dataloader)
    print("train acc %.5f" % accuracy)
    print("train avg_loss %.5f" % avg_loss)
    if train_large_loader is not None:
        print(f"use variance reduction large iterations @{vr_batch_numbers}")
    print("train batches: ", len(dataloader))

    et = time.time()
    return et - st, avg_loss, accuracy, losses


def train_drsom(dataloader, model, loss_fn, optimizer, logging_interval):
    st = time.time()
    size = len(dataloader.dataset)
    correct = 0
    total = 0
    avg_loss = 0
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        def closure(backward=True):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            if not backward:
                return loss
            if (
                optimizer.qpmode in {DRSOMModeQP.AutomaticDiff, DRSOMModeQP.FiniteDiff}
                or DRSOM_VERBOSE
            ):
                # only need for hvp
                loss.backward(create_graph=True)
            else:
                loss.backward()
            return loss

        # backpropagation

        loss = optimizer.step(closure=closure)
        avg_loss += loss.item()
        losses.append(loss.item())

        # compute prediction error
        outputs = model(X)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        if batch % logging_interval == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    accuracy = 100.0 * correct / total
    avg_loss = avg_loss / len(dataloader)
    print("train acc %.5f" % accuracy)
    print("train avg_loss %.5f" % avg_loss)
    print("train batches: ", len(dataloader))

    et = time.time()
    return et - st, avg_loss, accuracy, losses


def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    arr_pred = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            test_loss += loss_fn(outputs, y).item()
            arr_correct = predicted.eq(y)
            arr_pred.extend(arr_correct.tolist())
            correct += arr_correct.sum().item()
    test_loss /= num_batches
    correct /= size
    rstring = (
        f"Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    print(rstring)
    df = pd.DataFrame({"y": dataloader.dataset.targets, "yh": arr_pred})
    dfa = df.groupby("y").agg({"yh": "count"})
    dfb = df.groupby("y").agg({"yh": sum})
    dfg = pd.DataFrame({"total": dfa["yh"], "correct": dfb["yh"]}).assign(
        ratio=lambda df: df["correct"] / df["total"]
    )
    result = {"acc": (100 * correct), "avg_loss": test_loss, "df": dfg}
    return result


def adjust_lr(lr, epochs, epoch, optimizer, name=None):
    import bisect
    index = bisect.bisect_right(epochs, epoch)
    lr_now = lr / (5 ** index)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_now
    print(f"lr now: {lr_now}")


## SGD and NSGD
def parse_algo(method, model, **kwargs):
    """
    args: a string containing algorithm type and paras
        sgd
        sgd_clip([layer])
        nsgd([layer])
        adagrad([element], [layer])
        qhm_clip([layer])
    Note: if [layer], grad normalization is applied layerwise.
        Here every submodel with direct parameters is considered a layer.
    """
    from first_order.alg import Algorithm, SGD, NormalizedSGD, SGDClip

    net_paras = model.parameters()
    print(f"used kwargs: {kwargs}")
    if "nsgd" == method.lower():
        algo = NormalizedSGD
        para = ("lr", "momentum", "vr")
        return Algorithm(net_paras, algo, **{key: kwargs[key] for key in para})

    elif "sgd_clip" == method.lower():
        """
        g = momentum * g + (1 - momentum) * grad
        x = x - min(lr, gamma / |g|) * g
        """
        algo = SGDClip
        para = ("lr", "momentum", "gamma", "vr")
        return Algorithm(net_paras, algo, **{key: kwargs[key] for key in para})

    elif "sgd" == method.lower():
        from torch import optim
        return optim.SGD(
            net_paras,
            kwargs["lr"],
            momentum=kwargs["momentum"],
        )
    else:
        raise NotImplementedError

def main(args):
    print(f"Using [{device}] for the work")
    print(f"results saving to {args.save_path}")
    args.gamma = args.mu  # we use μ to avoid misleading parameterization.
    print("#" * 80)
    pprint(args.__dict__)
    print("#" * 80)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # transforms.Normalize(mean=[0.5], std=[0.5])
        ]
    )
    if args.dataset.lower() == "fmnist":
        train_data = datasets.FashionMNIST(
            root="./data/", transform=transform, train=True, download=True
        )

        test_data = datasets.FashionMNIST(
            root="./data/", transform=transform, train=False
        )
    else:
        train_data = datasets.MNIST(
            root="./data/", transform=transform, train=True, download=True
        )

        test_data = datasets.MNIST(root="./data/", transform=transform, train=False)

    if args.imbalance > 0:
        print("create an imbalanced dataset")
        train_data, imbalance_ratio = utils.imbalance_sampling(train_data)
    else:
        imbalance_ratio = None

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    print("#" * 80)
    if args.vr_ratio > 0:
        large_batch_size = int(round(args.batch_size * args.vr_batch_ratio))
        print(
            f"use variance reduction\n"
            f" @large batch @{large_batch_size}"
            f" @frequency @{args.vr_ratio}"
        )
        train_large_loader = DataLoader(
            train_data, batch_size=large_batch_size, shuffle=True
        )
    else:
        print(f"no variance reduction\n")
        train_large_loader = None

    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # define model
    if args.model == "log":
        input_dim = train_data.data.shape[1:]
        model = LogisticRegression(
            reduce(lambda x, y: x * y, input_dim), len(test_data.targets.unique())
        )
    else:
        model = CNNModel().to(device)

    if args.lossfunc == "dro":
        loss_fn = lambda yh, y: DRO_cross_entropy(yh, y, lbda=0.1)
        print("use dro loss as the target !")
    elif args.lossfunc == "dro-cvar":
        loss_fn = lambda yh, y: CVaR_cross_entropy(yh, y, lbda=1.0)
        print("use dro cvar as the target !")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    if args.trained_model is not None:
        dict = torch.load(args.trained_model)
        model.load_state_dict(dict)
        print("model loaded successfully!")
    else:
        print("train from scratch!")
    model.to(device)

    if args.optim == "drsom":
        func_kwargs = render_args(args)
        optimizer = DRSOM(model.parameters(), gamma=args.gamma, **func_kwargs)
        print(optimizer.get_name())
    else:
        optimizer = parse_algo(
            args.optim,
            model,
            lr=args.lr,
            momentum=args.momentum,
            vr=args.vr_ratio > 0,
            gamma=args.gamma,
        )
    print("#" * 80)
    all_avg_loss = []
    all_train_acc = []
    all_test_acc = []

    #
    hash_str = json.dumps(args.__dict__, skipkeys=True).__hash__()
    pathname = f"{args.save_path}/{args.optim}-{hash_str}"
    # this is record json file, we us
    fo = open(f"{pathname}.json", "w")
    print(f"result file: {fo}")

    epoch_list = [15, 40]
    per_step_losses = []
    for i in range(args.epoch):
        print("#" * 40 + f" {i + 1} " + "#" * 40)
        adjust_lr(args.lr, epoch_list, i, optimizer, name=args.optim)
        model.train()

        _, avg_loss, acc, losses = train(
            train_loader,
            args.optim,
            model,
            loss_fn,
            optimizer,
            args,
            train_large_loader=train_large_loader,
        )
        ################
        # profile details
        ################
        if args.optim.startswith("drsom"):
            print("|--- DRSOM COMPUTATION STATS ---")
            stats = pd.DataFrame.from_dict(DRSOM_GLOBAL_PROFILE)
            stats["avg"] = stats["total"] / stats["count"]
            stats = stats.sort_values(by="total", ascending=False)
            print(stats.to_markdown())

        if i % 5 == 0:
            print(f"saving to results/optim_{args.optim}_epoch_{i + 1}.pth")
            save_model(model, args, "optim_{}_epoch_{}.pth".format(args.optim, i + 1))
        rt = test(test_loader, model, loss_fn)
        all_avg_loss.append(avg_loss)
        all_train_acc.append(acc)
        all_test_acc.append(rt["acc"])
        per_step_losses = [*per_step_losses, *losses]
        # an extra details log for plotting
        record = {
            "avg_train_loss": all_avg_loss[i],
            "avg_train_acc": all_train_acc[i],
            "avg_test_acc": all_test_acc[i],
            "per_class_total": rt["df"]["total"].tolist(),
            "per_class_correct": rt["df"]["correct"].tolist(),
            "per_class_success_rate": rt["df"]["ratio"].tolist(),
            "per_class_training": imbalance_ratio,
            "t": i,
            **args.__dict__,
        }
        if i == args.epoch - 1:
            record["per_step_losses"] = per_step_losses
        json_str = json.dumps(record, skipkeys=True)
        fo.write(json_str)
        fo.write("\n")
        fo.flush()

    # save final accuracy


if __name__ == "__main__":
    args = make_args()
    main(args)

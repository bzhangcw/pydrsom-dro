import torch
import csv
from torch.utils.data import DataLoader
import argparse
from my_model import CNNModel
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
        "--lossfunc",
        required=False,
        type=str,
        choices=["dro", "usual"],
        help="select loss function",
        default="dro",
    )
    parser.add_argument(
        "--save_path", default=r"./results/0730/",
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
    parser.add_argument("--epoch", default=30, type=int, help="epoch number")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument(
        "--trained_model", default=None, help="the path to the saved trained model"
    )
    parser.add_argument(
        "--interval", required=False, type=int, default=20, help="logging interval"
    )
    parser.add_argument("--lr", default=1e-3, type=float)
    
    parser.add_argument("--out_dim", default=1)
    ## penalty size
    parser.add_argument("--gamma", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--epoch_list", default="10,25,40", type=str)
    parser.add_argument("--run_id", default=1, type=int, help="repetition id")
    add_parser_options(parser)
    args = parser.parse_args()
    return args


def train(dataloader, name, model, loss_fn, optimizer, args):
    model.train()
    if name.startswith("drsom"):
        return train_drsom(dataloader, model, loss_fn, optimizer, args.interval)
    st = time.time()
    size = len(dataloader.dataset)

    correct = 0
    total = 0
    avg_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        def closure(backward=True):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            return loss

        # backpropagation

        loss = optimizer.step(closure=closure)
        avg_loss += loss.item()

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
    print("train batches: ", len(dataloader))

    et = time.time()
    return et - st, avg_loss, accuracy


def train_drsom(dataloader, model, loss_fn, optimizer, ninterval):
    st = time.time()
    size = len(dataloader.dataset)
    correct = 0
    total = 0
    avg_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        def closure(backward=True):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            if not backward:
                return loss
            if optimizer.qpmode in {DRSOMModeQP.AutomaticDiff, DRSOMModeQP.FiniteDiff} or DRSOM_VERBOSE:
                # only need for hvp
                loss.backward(create_graph=True)
            else:
                loss.backward()
            return loss

        # backpropagation

        loss = optimizer.step(closure=closure)
        avg_loss += loss.item()

        # compute prediction error
        outputs = model(X)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        if batch % ninterval == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    accuracy = 100.0 * correct / total
    avg_loss = avg_loss / len(dataloader)
    print("train acc %.5f" % accuracy)
    print("train avg_loss %.5f" % avg_loss)
    print("train batches: ", len(dataloader))

    et = time.time()
    return et - st, avg_loss, accuracy


def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            test_loss += loss_fn(outputs, y).item()
            correct += predicted.eq(y).sum().item()
    test_loss /= num_batches
    correct /= size
    rstring = (
        f"Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    print(rstring)
    result = {"acc": (100 * correct), "avg_loss": test_loss}
    return result


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
    from algorithm import Algorithm, SGD, NormalizedSGD, SGDClip

    net_paras = model.parameters()
    print(method)
    if "nsgd" == method.lower():
        algo = NormalizedSGD
        para = ("lr", "momentum")

    elif "sgd_clip" == method.lower():
        """
            g = momentum * g + (1 - momentum) * grad
            x = x - min(lr, gamma / |g|) * g
        """
        algo = SGDClip
        para = ("lr", "momentum", "gamma")

    elif "sgd" == method.lower():
        algo = SGD
        para = ("lr", "momentum")

    else:
        raise NotImplementedError
    return Algorithm(net_paras, algo, **{key: kwargs[key] for key in para})


def adjust_lr(lr, epochs, epoch, optimizer):
    import bisect

    index = bisect.bisect_right(epochs, epoch)
    lr_now = lr / (10 ** index)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_now


def main(args):
    print(f"Using [{device}] for the work")
    print(f"results saving to {args.save_path}")
    # model
    input_dim = 28 * 28
    # model = LogisticRegression(input_dim, 10)
    model = CNNModel().to(device)

    if args.lossfunc == "dro":
        loss_fn = lambda yh, y: DRO_cross_entropy(yh, y, lbda=0.1)
        print("use dro loss as the target !")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    if args.trained_model is not None:
        dict = torch.load(args.trained_model)
        model.load_state_dict(dict)
        print("model loaded successfully!")
    else:
        print("train from scratch!")
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ]
    )
    if args.dataset == "FMNIST":
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

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    if args.optim == "drsom":
        func_kwargs = render_args(args)
        optimizer = DRSOM(model.parameters(), gamma=args.gamma, **func_kwargs)
        print(optimizer.get_name())
    elif args.optim == "sgd_clip":
        args.gamma = args.mu  # we use μ to avoid misleading parameterization.
        optimizer = parse_algo(
            args.optim, model, lr=args.lr, momentum=args.momentum, gamma=args.gamma,
        )
    else:
        optimizer = parse_algo(args.optim, model, lr=args.lr, momentum=args.momentum)

    epoch_list = [int(epoch) for epoch in args.epoch_list.split(",")]

    all_avg_loss = []
    all_train_acc = []
    all_test_acc = []

    if args.optim == "drsom":
        pathname = (
            args.save_path
            + f"/{args.optim}_{args.batch_size}_{optimizer.mode}_{optimizer.qpmode}.csv.{args.run_id}"
        )
    elif args.optim == "sgd_clip":
        pathname = (
            args.save_path
            + f"/{args.optim}_{args.batch_size}_{args.lr}_{args.gamma}_{args.mu}.csv.{args.run_id}"
        )
    else:
        pathname = (
            args.save_path
            + f"/{args.optim}_{args.batch_size}_{args.lr}_{args.gamma}.csv.{args.run_id}"
        )
    print("#" * 80)
    print(args.__dict__)
    print("#" * 80)
    fo = open(f"{pathname}.json", "w")
    with open(pathname, "w", encoding="utf-8", newline="",) as file_obj:
        header = ["all_train_loss", "all_train_acc", "all_test_acc", "run_id"]
        # 创建writer对象
        writer = csv.writer(file_obj)
        # 写表头
        writer.writerow(header)
        # 一次写入多行
        for i in range(args.epoch):
            print(
                "-----------------------epoch {}-----------------------".format(i + 1)
            )
            if args.optim == "drsom":
                print(
                    "-----------current methods: drsom, initial region:{:.6f}-----------.".format(
                        args.gamma
                    )
                )
            if args.optim == "sgd":
                adjust_lr(args.lr, epoch_list, i, optimizer)
                print(
                    "-----------current methods: sgd, learning rate: {:.6f}-----------".format(
                        args.lr
                    )
                )
            if args.optim == "nsgd":
                adjust_lr(args.lr, epoch_list, i, optimizer)
                print(
                    "-----------current methods: nsgd, learning rate: {:.6f}-----------".format(
                        args.lr
                    )
                )
            model.train()
            # train_loss, train_acc = train_loop(
            #     model, train_loader, optimizer, DRO_cross_entropy, device,
            # )
            # with torch.no_grad():
            #     model.eval()
            #     val_acc = val_loop(model, test_loader, device)
            _, avg_loss, acc = train(
                train_loader, args.optim, model, loss_fn, optimizer, args
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
                save_model(
                    model, args, "optim_{}_epoch_{}.pth".format(args.optim, i + 1)
                )
            rt = test(test_loader, model, loss_fn)
            all_avg_loss.append(avg_loss)
            all_train_acc.append(acc)
            all_test_acc.append(rt["acc"])
            writer.writerow(
                (all_avg_loss[i], all_train_acc[i], all_test_acc[i], args.run_id)
            )
            file_obj.flush()

            # an extra details log for plotting
            record = {
                "avg_train_loss": all_avg_loss[i],
                "avg_train_acc": all_train_acc[i],
                "avg_test_acc": all_test_acc[i],
                "t": i,
                **args.__dict__,
            }

            import json

            json_str = json.dumps(record, skipkeys=True)
            fo.write(json_str)
            fo.write("\n")
            fo.flush()


if __name__ == "__main__":
    args = make_args()
    main(args)

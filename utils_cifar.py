import argparse

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pydrsom.drsom_utils import *
from pydrsom.drsom import DRSOMB as DRSOM
from pydrsom.drsom_utils import add_parser_options, DRSOMDecayRules
from cifar10.models import *
from utils import imbalance_sampling
from first_order.alg import *
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch CIFAR10 Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="resnet",
        type=str,
        help="model",
        choices=["resnet18", "resnet34"],
    )
    parser.add_argument(
        "--optim",
        default="sgd",
        type=str,
        help="optimizer",
        choices=["sgd1", "sgd2", "sgd3", "adam", "drsom", "nsgd", "sgd_clip"],
    )
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--batch", default=128, type=int, help="batch size")
    parser.add_argument(
        "--gamma", default=1e-3, type=float, help="for clip"
    )

    parser.add_argument(
        "--tflogger", default="/tmp/", type=str, help="tf logger directory"
    )
    ##############
    # sgd & adam
    ##############
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument("--ckpt_name", type=str, help="resume from checkpoint")
    parser.add_argument(
        "--epoch", "-e", default=200, type=int, help="num of epoches to run"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="weight decay for optimizers"
    )
    ##############
    # learning rate scheduler
    parser.add_argument(
        "--lrstep",
        default=80,
        type=int,
        help="step for learning rate scheduler\n"
        " - for Adam/SGD, it corresponds to usual step definition for the lr scheduler\n",
    )
    parser.add_argument(
        "--lrcoeff",
        default=1e-1,
        type=float,
        help="step for learning rate scheduler\n"
        " - for Adam/SGD, it corresponds to usual step definition for the lr scheduler\n",
    )
    ##############
    # drsom
    ##############
    add_parser_options(parser)
    return parser


def build_dataset(args):
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    if args.imbalance > 0:
        print("create an imbalanced dataset")
        trainset, imbalance_ratio = imbalance_sampling(trainset)
    else:
        imbalance_ratio = None

    train_loader = DataLoader(
        trainset, batch_size=args.batch, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch, shuffle=False, num_workers=2
    )

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


def build_model(args, device, ckpt=None):
    print("==> Building model..")
    net = {"resnet18": ResNet18, "resnet34": ResNet34}[args.model]()
    print(f"==> Building model {args.model}")
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt["net"])

    return net


def create_optimizer(args, model, start_epoch=0):
    model_params = model.parameters()
    if args.optim == "sgd1":
        return optim.SGD(
            model_params,
            args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    if args.optim == "sgd2":
        return optim.SGD(
            model_params,
            args.lr,
            momentum=0.95,
            weight_decay=args.weight_decay,
        )
    if args.optim == "sgd3":
        return optim.SGD(
            model_params,
            args.lr,
            momentum=0.99,
            weight_decay=args.weight_decay,
        )
   
    # my second-order method
    elif args.optim == "drsom":
        return DRSOM(model_params, **render_args(args))
    else:
        print("using clipping algorithms")
        from first_order.alg import Algorithm, NormalizedSGD, SGDClip

        if "nsgd" == args.optim.lower():
            algo = NormalizedSGD
            para = ["lr"]

        elif "sgd_clip" in args.optim.lower():
            """
            g = momentum * g + (1 - momentum) * grad
            x = x - min(lr, gamma / |g|) * g
        """
            algo = SGDClip
            para = ["lr", "gamma"]

        else:
            raise NotImplementedError

        alg_instance = Algorithm(
            model_params, algo, momentum=0.9, **{key: args.__dict__[key] for key in para}
        )
        print(alg_instance)
        return alg_instance


def get_ckpt_name(model="resnet", optimizer="sgd", lr=0.1):
    """
  get checkpoint name for optimizers except for DRSOM.
  Args:
    model:
    optimizer:
    lr:
    final_lr:
    momentum:
    beta1:
    beta2:
    gamma:

  Returns:

  """
    name = {
        "sgd1": "lr{}".format(lr),
        "sgd2": "lr{}".format(lr),
        "sgd3": "lr{}".format(lr),
        "adam": "lr{}".format(lr),
        "sgd_clip": "lr{}".format(lr),
        "nsgd": "lr{}".format(lr),
    }[optimizer]
    return "[{}]-{}-{}".format(model, optimizer, name)


def train(net, epoch, device, data_loader, name, optimizer, criterion):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    size = len(data_loader.dataset)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        def closure(backward=True):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if not backward:
                return loss
            if name.startswith("drsom"):
                if optimizer.qpmode in {DRSOMModeQP.AutomaticDiff, DRSOMModeQP.FiniteDiff}:
                    # only need for hvp
                    loss.backward(create_graph=True)
                else:
                    loss.backward()
            else:
                loss.backward()
            return loss

        loss = optimizer.step(closure=closure)
        outputs = net(inputs)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 20 == 0:
            loss, current = loss.item(), batch_idx * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    accuracy = 100.0 * correct / total
    print("train acc  %.3f" % accuracy)
    print("train loss %.3f" % (train_loss / len(data_loader)))

    return accuracy, train_loss / len(data_loader)


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    arr_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            arr_correct = predicted.eq(targets)
            correct += arr_correct.sum().item()
            arr_pred.extend(arr_correct.tolist())

    accuracy = 100.0 * correct / total
    print(" test acc  %.3f" % accuracy)
    print(" test loss %.3f" % (test_loss / len(data_loader)))
    df = pd.DataFrame(
        {
            "y": data_loader.dataset.targets,
            "yh": arr_pred
        }
        )
    dfa = df.groupby("y").agg(
        {
            "yh": "count"
        }
        )
    dfb = df.groupby("y").agg(
        {
            "yh": sum
        }
        )
    dfg = pd.DataFrame(
        {
            "total": dfa["yh"],
            "correct": dfb["yh"]
        }
        ).assign(
        ratio=lambda df: df["correct"] / df["total"]
    )
    result = {
        "acc": (100 * correct),
        "avg_loss": test_loss,
        "df": dfg
    }
    return accuracy, test_loss / len(data_loader)

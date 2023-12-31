"""
A script for DRSOM on CIFAR dataset.
@author: Chuwen Zhang
@note:
  This script runs DRSOM and compares to Adam, SGD, and so forth.
  ################################################################
    usage:
      $ python main.py -h
  ################################################################
"""

from __future__ import print_function

import json

from utils_cifar import *
from utils import DRO_cross_entropy, CVaR_cross_entropy


def main():
    parser = get_parser()
    parser.add_argument(
        "--lossfunc",
        required=False,
        type=str,
        choices=["dro", "dro-cvar", "usual"],
        help="select loss function",
        default="dro",
    )
    parser.add_argument(
        "--imbalance",
        required=False,
        type=int,
        help="whether to sample an imbalanced data, default true ",
        default=1,
    )

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))
    train_loader, test_loader = build_dataset(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume:
        ckpt = load_checkpoint(args.ckpt_name)
        start_epoch = ckpt["epoch"] + 1
    else:
        ckpt = None
        start_epoch = 0

    net = build_model(args, device, ckpt=ckpt)
    # criterion = nn.CrossEntropyLoss()
    if args.lossfunc == "dro":
        criterion = lambda yh, y: DRO_cross_entropy(yh, y, lbda=0.1)
        print("use dro loss as the target !")
    elif args.lossfunc == "dro-cvar":
        criterion = lambda yh, y: CVaR_cross_entropy(yh, y, lbda=1.0)
        print("use dro cvar as the target !")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, net, start_epoch=start_epoch)

    if args.optim.startswith("drsom"):
        # get a scheduler
        pass
    else:
        # get a scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lrstep,
            gamma=args.lrcoeff,
        )

    if args.optim.startswith("drsom"):
        log_name = (
                f"[{args.model}]" + "-" + query_name(optimizer, args.optim, args, ckpt)
        )
    else:
        log_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr)
    print(f"Using model: {args.model}")
    print(f"Using optimizer:\n {log_name}")

    writer = SummaryWriter(log_dir=os.path.join(args.tflogger, log_name))
    epoch_intervals = list(range(start_epoch, start_epoch + args.epoch))
    for epoch in epoch_intervals:
        try:
            if args.optim.startswith("drsom"):
                pass

            else:
                scheduler.step()
                print(f"lr scheduler steps: {scheduler.get_lr()}")

            (
                train_acc,
                train_loss,
            ) = train(
                net, epoch, device, train_loader, args.optim, optimizer, criterion
            )
            test_acc, test_loss = test(net, device, test_loader, criterion)

            # writer
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/test", test_acc, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)

            # Save checkpoint.
            if epoch % 5 == 0:
                print("Saving..")

                state = {
                    "net": net.state_dict(),
                    "acc": test_acc,
                    "epoch": epoch,
                }
                if not os.path.isdir("checkpoint"):
                    os.mkdir("checkpoint")

                torch.save(state, os.path.join("checkpoint", f"{log_name}-{epoch}"))

                ######################################
                # train_accuracies.append(train_acc)
                # test_accuracies.append(test_acc)
                # if not os.path.isdir('curve'):
                #   os.mkdir('curve')
                # torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                #            os.path.join('curve', ckpt_name))
                #######################################

        except KeyboardInterrupt:
            print(f"Exiting at {epoch}")
            break
        ################
        # profile details
        ################
        if args.optim.startswith("drsom"):
            import pandas as pd

            print("|--- DRSOM COMPUTATION STATS ---")
            stats = pd.DataFrame.from_dict(DRSOM_GLOBAL_PROFILE)
            stats["avg"] = stats["total"] / stats["count"]
            stats = stats.sort_values(by="total", ascending=False)
            print(stats.to_markdown())

    test_acc, test_loss = test(net, device, test_loader, criterion)

if __name__ == "__main__":
    main()

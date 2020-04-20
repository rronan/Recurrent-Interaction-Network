import torch

from cnr.model import CNR
from cnr.parser import parse_args
from pyronan.dataset import make_loader
from pyronan.model import make_model
from pyronan.train import Trainer
from pyronan.utils.torchutil import set_seed


def collate_fn(batch):
    res_x = []
    res_y = []
    for x, y in batch:
        res_x.append(torch.FloatTensor(x))
        res_y.append(torch.LongTensor(y))
    return res_x, torch.stack(res_y)


def train(args):
    set_seed(args.seed, args.gpu)
    loader_dict = {
        set_: make_loader(
            Dataset,
            args,
            set_,
            args.batch_size,
            args.num_workers,
            args.pin_memory,
            collate_fn=collate_fn,
        )
        for set_ in ["train", "val"]
    }
    model = make_model(CNR, args, args.gpu, args.data_parallel, args.load)
    trainer = Trainer(model, args)
    trainer(loader_dict, args.train_epochs)


def main():
    args, _ = parse_args(None)
    print(args)
    train(args)


if __name__ == "__main__":
    main()

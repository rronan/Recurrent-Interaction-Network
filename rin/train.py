from torch.utils.data import DataLoader

from datasets import Dataset_pybullet, Dataset_ue
from pyronan.model import make_model
from pyronan.train import Trainer
from pyronan.utils.torchutil import set_seed
from rin.intnet import IntNet
from rin.parser import parse_args
from rin.recurrent import Recurrent
from rin.sequence import Sequence

DATASETS = {"ue": Dataset_ue, "pybullet": Dataset_pybullet}


def make_loader(args, bsz, num_workers):
    dataset_dict, loader_dict = {}, {}
    for set_ in ["train", "val"]:
        dataset = DATASETS[args.dataset](args, set_)
        dataset_dict[set_] = Sequence(
            dataset, args.order, args.seq_length, args.logvar_bias, args.bidirectional
        )
        loader_dict[set_] = DataLoader(dataset, bsz, num_workers=num_workers)
    return dataset_dict, loader_dict


def train(args):
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed, args.gpu)
    dataset_dict, loader_dict = make_loader(args, args.bsz, args.num_workers)
    intnet = make_model(IntNet, args, args.gpu, args.data_parallel, args.load)
    model = Recurrent(intnet.core_module, args)
    trainer = Trainer(model, args)
    trainer.train(args.train_epochs, args.verbose)


def main():
    args = parse_args()
    print(args)
    train(args)


if __name__ == "__main__":
    main()

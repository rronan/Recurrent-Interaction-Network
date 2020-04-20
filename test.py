from argparse import ArgumentParser
from pathlib import Path

from pyronan.utils.misc import append_timestamp
from Recurrent_Interaction_Networks.rollout.intphys import intphys


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument(
        "task", nargs="*", choices=["compute", "draw", "eval", "sort", []]
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--input_prefix", type=Path, default="/sata/rriochet/intphys2019/test"
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--name", type=append_timestamp, default="")
    parser.add_argument(
        "--outdir",
        type=Path,
        default="/sequoia/data1/rriochet/Recurrent_Interaction_Networks/intphys",
    )
    parser.add_argument("--intnet", default="191107_214043_sweep_intnet_ue_0")
    parser.add_argument("--prn", default="191112_011454_sweep_prn_ue_3")
    parser.add_argument("--idx_start", type=int, default=0)
    parser.add_argument("--idx_stop", type=int, default=32)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--n_iter", type=int, default=2000)
    parser.add_argument("--ghk", nargs="+", type=float, default=[0.8, 0.1])
    parser.add_argument("--nll_weight", nargs="+", type=float, default=[1, 4, 4, 4, 1])
    parser.add_argument("--lambda_", type=float, default=1)
    parser.add_argument("--step_list", nargs="+", type=int, default=[-1, 1])
    parser.add_argument("--nc_out", type=int, default=5)
    parser.add_argument("--video_slice", default=slice(None))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "/sequoia/data1/rriochet/Positional_Reconstruction_Network/datasets/metadata.json"
        ),
    )
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--movement", default=None)
    parser.add_argument("--nobjects", type=int, default=None)
    parser.add_argument("--hw", type=int, default=128)
    parser.add_argument("--block", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--item_list", nargs="+", default=[])
    args = parser.parse_args(argv)
    return args


def main():
    args = parse_args()
    print(args)
    intphys(args)


if __name__ == "__main__":
    main()

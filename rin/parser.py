from argparse import ArgumentParser
from pathlib import Path

from pyronan.model import parser_model
from pyronan.train import parser_train
from pyronan.utils.misc import parse_slice
from pyronan.utils.parser import parser_base


def parse_args(argv=None):
    parser = ArgumentParser(parents=[parser_base, parser_train, parser_model])
    parser.add_argument("--features_prefix", type=Path, default=None)
    parser.add_argument(
        "--input_prefix",
        type=Path,
        default="/sequoia/data1/rriochet/hybrid_data/bullet_obj_fps30/raw",
    )
    parser.add_argument("--dataset", default="Dataset_pybullet")
    parser.add_argument("--clip", type=float, default=None, help="clip input")
    parser.add_argument("--remove_to_depth", type=float, default=None)
    parser.add_argument("--video_slice", type=parse_slice, default=slice(None))
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random_camera", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seq_length", type=int, default=2)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--ghk", type=float, nargs="+")
    parser.add_argument("--logvar_bias", type=float, default=None)
    parser.add_argument("--item_list", nargs="+")
    parser.add_argument("--select_features", type=int, nargs="+")
    parser.add_argument("--weight_gauss", default=None, type=float)
    parser.add_argument("--loss_weight_list", default=None, type=float, nargs="+")
    parser.add_argument("--weight_occluder", type=float, default=None)
    parser.add_argument("--n_occluder", type=float, default=3)
    parser.add_argument("--init_scale", type=float, default=1)
    parser.add_argument("--D_s", type=int, default=9, help="The State Dimension")
    parser.add_argument("--N_o", type=int, default=8, help="The Number of Objects")
    parser.add_argument("--D_r", type=int, default=1, help="The Relationship Dimension")
    parser.add_argument(
        "--D_x", type=int, default=1, help="The External Effect Dimension"
    )
    parser.add_argument("--D_e", type=int, default=50, help="The Effect Dimension")
    parser.add_argument(
        "--D_p", type=int, default=3, help="The Object Modeling Output Dimension"
    )
    parser.add_argument("--d_r", type=int, default=150)
    parser.add_argument("--d_x", type=int, default=150)
    parser.add_argument("--d_f", type=int, default=100)
    parser.add_argument("--n_r", type=int, default=4)
    parser.add_argument("--n_x", type=int, default=4)
    parser.add_argument("--n_f", type=int, default=2)
    # frame related
    parser.add_argument("--hw", type=int, default=None)
    parser.add_argument("--nc_out", type=int, default=None)
    parser.add_argument("--draw_occluder", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)
    args.checkpoint /= args.name
    if args.loss_weight_list is None:
        args.loss_weight_list = [1] * args.seq_length
    type_dict = {x.dest: x.type for x in parser._actions}
    return args, type_dict

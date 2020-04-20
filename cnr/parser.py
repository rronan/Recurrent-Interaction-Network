from argparse import ArgumentParser
from pathlib import Path

from pyronan.model import parser_model
from pyronan.train import parser_train
from pyronan.utils.misc import parse_slice
from pyronan.utils.parser import parser as common


def parse_args(argv=None):
    parser = ArgumentParser(parents=[common, parser_model, parser_train])
    parser.set_defaults(
        model="PRN",
        checkpoint=Path(
            "/sequoia/data1/rriochet/Positional_Reconstruction_Network/checkpoints"
        ),
        batch_size=8,
    )
    parser.add_argument("--model", default="PRN", help="model to train")
    parser.add_argument("--upsample_mode", default="bilinear")
    parser.add_argument("--dataset", default="Dataset_carla")
    parser.add_argument(
        "--input_prefix",
        type=Path,
        default=Path("/sequoia/data1/rriochet/carla_dataset/first_iteration"),
    )
    parser.add_argument("--item_list", nargs="+", default=[])
    parser.add_argument("--height", type=int, default=135)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--length_polygon", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n_obj", type=int, default=100, help="number of objects max")
    parser.add_argument("--nc_in", type=int, default=None)
    parser.add_argument("--nc_out", type=int, default=None)
    parser.add_argument("--nf", type=int, default=32)
    parser.add_argument(
        "--up_conv",
        default="UpSamplingConvolution",
        choices=[
            "SpatialFullConvolution",
            "UpSamplingConvolution",
            "SubPixelConvolution",
        ],
    )
    parser.add_argument("--remove_to_depth", type=float, default=None)
    parser.add_argument("--relative_depth", action="store_true")
    parser.add_argument("--depth_noise", type=float, default=None)
    parser.add_argument("--input_slice", type=parse_slice, default=slice(None))
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--mse", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--depth_trunc", action="store_true")
    parser.add_argument("--depth_weight", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=None)
    args = parser.parse_args(argv)
    args.nc_in = args.length_polygon * 2 + args.n_obj + 1
    args.nc_out = args.n_obj
    args.latent_dim = args.nf * 4
    args.checkpoint.mkdir(exist_ok=True)
    args.checkpoint /= args.name
    type_dict = {x.dest: x.type for x in parser._actions}
    return args, type_dict

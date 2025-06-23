import os
import sys

from dinov2.eval.linear import get_args_parser, main

if __name__ == "__main__":
    description = "Linear Evaluation"
    args_parser = get_args_parser(description=description)
    args_parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use a balanced sampler for training data",
    )
    args_parser.add_argument(
        "--balanced-sampler-mode",
        type=lambda x: int(x) if x.isdigit() else x,
        default="downsampling",
        help="Balanced sampler mode. Can be 'downsampling', 'upsampling' or an integer value",
    )
    args_parser.add_argument(
        "--logit-adjusted-loss",
        action="store_true",
        help="Use logit adjusted loss",
    )
    args_parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of classes for classification",
    )
    args_parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs",
    )
    args_parser.add_argument(
        "--total-batch-size",
        type=int,
        default=256,
        help="Total batch size across all GPUs",
    )
    args = args_parser.parse_args()

    sys.exit(main(args))

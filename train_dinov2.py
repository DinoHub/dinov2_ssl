from dinov2.train import get_args_parser, main

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

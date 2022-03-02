import argparse

from analyze_tablegen.extraction import *


arg_parser = argparse.ArgumentParser(description='Print IRDL files extracted from TableGen')
arg_parser.add_argument("--dialect",
                        type=str,
                        nargs="?",
                        required=False,
                        help="Name of the dialect to print")

arg_parser.add_argument("--input-file",
                        type=str,
                        nargs="?",
                        required=False,
                        help="Path to JSON input file")

if __name__ == "__main__":
    args = arg_parser.parse_args()
    f = open(args.input_file, "r")
    stats = get_stats_from_json(f.read())
    if args.dialect not in stats.dialects:
        raise Exception(f"Dialect {args.dialect} does not exist, the available dialects are [{', '.join(stats.dialects.keys())}]")
    print(stats.dialects[args.dialect].as_str())

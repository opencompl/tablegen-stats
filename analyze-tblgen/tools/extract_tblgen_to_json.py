import argparse

from analyze_tablegen.extraction import *


arg_parser = argparse.ArgumentParser(description='Extract TableGen files into json format')
arg_parser.add_argument("--llvm-dir",
                        type=str,
                        nargs="?",
                        required=True,
                        help="Path to llvm project directory")

arg_parser.add_argument("--tblgen-extract-path",
                        type=str,
                        nargs="?",
                        required=True,
                        help="Path to tblgen-extract")

arg_parser.add_argument("--output-file",
                        type=str,
                        nargs="?",
                        required=True,
                        help="Path to JSON output file")

if __name__ == "__main__":
    args = arg_parser.parse_args()
    res = get_files_contents_as_json(args.llvm_dir, args.tblgen_extract_path)
    f = open(args.output_file, "w")
    f.write(res)
    f.close()

import argparse

from analyze_tablegen.extraction import *


def write_test_case(path: str, dialect: Dialect, irdl_opt_compatible):
    if irdl_opt_compatible:
        f = open(path + dialect.name + ".irdl", "w")
        f.write(dialect.as_str())
        f.close()
    else:
        f = open(path + "/irdl-opt-parseable/" + dialect.name + ".irdl", "w")
        f.write(dialect.as_str(current_irdl_support=True))
        f.close()


def write_all_test_cases(json_path: str, irdl_directory_path: str):
    f = open(json_path, "r")
    stats = get_stats_from_json(f.read())
    for _, dialect in stats.dialects.items():
        write_test_case(irdl_directory_path, dialect, True)
        write_test_case(irdl_directory_path, dialect, False)


arg_parser = argparse.ArgumentParser(description='print irdl test cases')
arg_parser.add_argument("--input-file",
                        type=str,
                        nargs="?",
                        default="mlir_dialects.json",
                        help="Path to input file containing all dialects")

arg_parser.add_argument("--output-dir",
                        type=str,
                        nargs="?",
                        default="examples/mlir_dialects/",
                        help="Path to output directory")

if __name__ == "__main__":
    args = arg_parser.parse_args()
    write_all_test_cases(args.input_file, args.output_dir)

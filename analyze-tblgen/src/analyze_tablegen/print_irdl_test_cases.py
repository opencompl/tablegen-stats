import argparse

from analyze_tablegen.irdl import *
from analyze_tablegen.extraction import *


def get_test_case(dialect: Dialect) -> str:
    header = "// RUN: dyn-opt %s | dyn-opt | FileCheck %s\n"
    code = dialect.as_str()
    test = code.split("\n")
    test = ["// CHECK: " + test_line for test_line in test if len(test_line) != 0]
    test = "\n".join(test)
    return header + code + "\n" + test


def write_test_case(path: str, dialect: Dialect):
    f = open(path + dialect.name + ".irdl", "w")
    f.write(get_test_case(dialect))
    f.close()


def write_all_test_cases(json_path: str, irdl_directory_path: str):
    f = open(json_path, "r")
    stats = get_stats_from_json(f.read())
    for _, dialect in stats.dialects.items():
        write_test_case(irdl_directory_path, dialect)


arg_parser = argparse.ArgumentParser(description='print irdl test cases')
arg_parser.add_argument("--input-file",
                        type=str,
                        nargs="?",
                        required=True,
                        help="Path to input file containing all dialects")

arg_parser.add_argument("--test-dir",
                        type=str,
                        nargs="?",
                        required=True,
                        help="Path to IRDL test directory")

if __name__ == "__main__":
    args = arg_parser.parse_args()
    write_all_test_cases(args.input_file, args.test_dir)

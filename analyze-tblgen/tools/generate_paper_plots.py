import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import datetime as dt

from analyze_tablegen.extraction import *

col0 = [240/255, 249/255, 232/255, 1]
col1 = [186/255, 228/255, 188/255, 1]
col2 = [123/255, 204/255, 196/255, 1]
col3 = [43/255, 140/255, 190/255, 1]
col4 = [8/255, 104/255, 172/255, 1]


light_blue = [166/255, 206/255, 227/255, 1]
dark_blue = [15/255, 90/255, 160/255, 1]
black = [0.2, 0.2, 0.2, 1]
mid_blue = [0.6 * light_blue[i] + 0.4 * dark_blue[i] for i in range(4)]
mid_light_blue = [0.8 * light_blue[i] + 0.2 * dark_blue[i] for i in range(4)]
mid_dark_blue = [0.3 * light_blue[i] + 0.7 * dark_blue[i] for i in range(4)]
light_green = [178/255, 223/255, 138/255, 1]
dark_green = [51/255, 160/255, 44/255, 1]
mid_green = [0.6 * light_green[i] + 0.4 * dark_green[i] for i in range(4)]


light_green = col1
dark_blue = col3

type_params_decl = dict()
attr_params_decl = dict()

type_verifier = dict()
attr_verifier = dict()

type_param_groups = dict()
attr_param_groups = dict()

num_operands_mean = list()
num_operands_per_dialect = dict()
num_var_operands_mean = list()
num_var_operands_per_dialect = dict()
num_results_mean = list()
num_results_per_dialect = dict()
num_var_results_mean = list()
num_var_results_per_dialect = dict()
num_attributes_mean = list()
num_attributes_per_dialect = dict()
num_regions_mean = list()
num_regions_per_dialect = dict()
has_verifier_mean = list()
has_verifier_per_dialect = dict()
op_args_decl_mean = list()
op_args_decl_per_dialect = dict()
op_non_decl_constraints = dict()


def get_op_list_distribution(ops: List[Op],
                             f: Callable[[Op], int],
                             maxi: Optional[int] = None) -> List[float]:
    res = [f(op) for op in ops]
    if maxi is None:
        maxi = max(res)
    return [res.count(i) for i in range(maxi + 1)]


def get_global_op_distribution(stats: Stats, f: Callable[[Op], int]):
    global_res = get_op_list_distribution(stats.ops, f)
    maxi = len(global_res) - 1
    per_dialect_res = dict()
    for dialect_name, dialect in stats.dialects.items():
        per_dialect_res[dialect_name] = get_op_list_distribution(list(
            dialect.ops.values()),
                                                                 f,
                                                                 maxi=maxi)
    return global_res, per_dialect_res


def get_attr_list_distribution(attrs: List[Attr], f: Callable[[Attr], int], maxi: Optional[int] = None) -> \
        List[
            float]:
    res = [f(attr) for attr in attrs]
    if maxi is None:
        maxi = max(res)
    return [res.count(i) for i in range(maxi + 1)]


def get_global_attr_distribution(stats: Stats, f: Callable[[Attr], int]):
    global_res = get_attr_list_distribution(stats.attrs, f)
    maxi = len(global_res) - 1
    per_dialect_res = dict()
    for dialect_name, dialect in stats.dialects.items():
        per_dialect_res[dialect_name] = get_attr_list_distribution(list(
            dialect.attrs.values()),
                                                                   f,
                                                                   maxi=maxi)
    return global_res, per_dialect_res


def get_type_list_distribution(types: List[Type], f: Callable[[Type], int], maxi: Optional[int] = None) -> \
        List[
            float]:
    res = [f(typ) for typ in types]
    if maxi is None:
        maxi = max(res)
    return [res.count(i) for i in range(maxi + 1)]


def get_global_type_distribution(stats: Stats, f: Callable[[Type], int]):
    global_res = get_type_list_distribution(stats.types, f)
    maxi = len(global_res) - 1
    per_dialect_res = dict()
    for dialect_name, dialect in stats.dialects.items():
        per_dialect_res[dialect_name] = get_type_list_distribution(list(
            dialect.types.values()),
                                                                   f,
                                                                   maxi=maxi)
    return global_res, per_dialect_res


T = TypeVar("T")


def get_dialect_values(stats: Stats, f: Callable[[Dialect],
                                                 T]) -> Dict[str, T]:
    return {key: f(value) for key, value in stats.dialects.items()}


def gather_type_parameters_decl_data(stats: Stats):
    type_res = get_global_type_distribution(
        stats,
        lambda typ: int(all([p.is_declarative()
                             for p in typ.parameters])))[1]
    attr_res = get_global_attr_distribution(
        stats,
        lambda attr: int(all([p.is_declarative()
                              for p in attr.parameters])))[1]
    global type_params_decl
    global attr_params_decl
    # noinspection PyTypeChecker
    type_params_decl = dict(filter(lambda x: sum(x[1]) != 0, type_res.items()))
    # noinspection PyTypeChecker
    attr_params_decl = dict(filter(lambda x: sum(x[1]) != 0, attr_res.items()))


def gather_type_verifiers_data(stats: Stats):
    type_res = get_global_type_distribution(
        stats, lambda typ: int(typ.hasVerifier))[1]
    attr_res = get_global_attr_distribution(
        stats, lambda attr: int(attr.hasVerifier))[1]
    global type_verifier
    global attr_verifier
    # noinspection PyTypeChecker
    type_verifier = dict(filter(lambda x: sum(x[1]) != 0, type_res.items()))
    # noinspection PyTypeChecker
    attr_verifier = dict(filter(lambda x: sum(x[1]) != 0, attr_res.items()))


def gather_type_parameters_type_data(stats: Stats):
    distr_typ = dict()
    for typ in stats.types:
        for param in typ.parameters:
            distr_typ.setdefault(param.get_group(), 0)
            distr_typ[param.get_group()] += 1
    distr_attr = dict()
    for attr in stats.attrs:
        for param in attr.parameters:
            distr_attr.setdefault(param.get_group(), 0)
            distr_attr[param.get_group()] += 1

    def distr_to_group(distr):
        group_distr = dict()
        for elem, val in distr.items():
            if elem[-5:] == "array":
                group_distr.setdefault(elem[:-6], [0, 0])
                group_distr[elem[:-6]][1] = val
            else:
                group_distr.setdefault(elem, [0, 0])
                group_distr[elem][0] = val
        return group_distr

    global type_param_groups
    global attr_param_groups
    type_param_groups = distr_to_group(distr_typ)
    attr_param_groups = distr_to_group(distr_attr)


def get_culprit_constraint_group(constr: PredicateConstraint) -> str:
    if constr.predicate == "isStrided($_self.cast<::mlir::MemRefType>())":
        return "stride check"

    if constr.predicate == "$_self.cast<::mlir::IntegerAttr>().getInt() >= 0" or \
            constr.predicate == "$_self.cast<::mlir::IntegerAttr>().getInt() <= 3" or \
            constr.predicate == "$_self.cast<IntegerAttr>().getValue().isStrictlyPositive()" or \
            constr.predicate == "$_self.cast<::mlir::IntegerAttr>().getValue().isNegative()":
        return "integer inequality"

    if constr.predicate == "$_self.cast<::mlir::LLVM::LLVMStructType>().isOpaque()":
        return "struct opacity"

    assert False


def get_constraints_culprits(stats: Stats) -> Dict[Constraint, int]:
    culprits = dict()

    def add_constraint(constraint: Constraint):
        if isinstance(constraint,
                      PredicateConstraint) and not constraint.is_declarative():
            group = get_culprit_constraint_group(constraint)
            culprits.setdefault(group, 0)
            culprits[group] += 1

    stats.walk_constraints(add_constraint)
    return culprits


def gather_op_stats_data(stats: Stats):
    num_operands = get_global_op_distribution(stats, lambda x: len(x.operands))
    global num_operands_mean
    num_operands_mean = num_operands[0]
    global num_operands_per_dialect
    num_operands_per_dialect = num_operands[1]

    num_var_operands = get_global_op_distribution(
        stats, lambda x: sum([
            isinstance(operand.constraint, VariadicConstraint)
            for operand in x.operands
        ]))
    global num_var_operands_mean
    num_var_operands_mean = num_var_operands[0]
    global num_var_operands_per_dialect
    num_var_operands_per_dialect = num_var_operands[1]

    num_results = get_global_op_distribution(stats, lambda x: len(x.results))
    global num_results_mean
    num_results_mean = num_results[0]
    global num_results_per_dialect
    num_results_per_dialect = num_results[1]

    num_var_results = get_global_op_distribution(
        stats, lambda x: sum([
            isinstance(result.constraint, VariadicConstraint)
            for result in x.results
        ]))
    global num_var_results_mean
    num_var_results_mean = num_var_results[0]
    global num_var_results_per_dialect
    num_var_results_per_dialect = num_var_results[1]

    num_attributes = get_global_op_distribution(stats,
                                                lambda x: len(x.attributes))
    global num_attributes_mean
    num_attributes_mean = num_attributes[0]
    global num_attributes_per_dialect
    num_attributes_per_dialect = num_attributes[1]

    num_regions = get_global_op_distribution(stats, lambda x: len(x.regions))
    global num_regions_mean
    num_regions_mean = num_regions[0]
    global num_regions_per_dialect
    num_regions_per_dialect = num_regions[1]

    has_verifiers = get_global_op_distribution(stats,
                                               lambda x: int(x.hasVerifier))
    global has_verifier_mean
    has_verifier_mean = has_verifiers[0]
    global has_verifier_per_dialect
    has_verifier_per_dialect = has_verifiers[1]

    op_args_decl = get_global_op_distribution(
        stats, lambda x: int(x.is_operands_results_attrs_declarative()))
    global op_args_decl_mean
    op_args_decl_mean = op_args_decl[0]
    global op_args_decl_per_dialect
    op_args_decl_per_dialect = op_args_decl[1]
    global op_non_decl_constraints
    op_non_decl_constraints = get_constraints_culprits(stats)


def gather_all_plots_data(stats: Stats):
    gather_type_parameters_decl_data(stats)
    gather_type_verifiers_data(stats)
    gather_type_parameters_type_data(stats)
    gather_op_stats_data(stats)


def print_data_used_in_text():
    print("-" * 60)
    print(25 * " " + "Operations")
    print("-" * 60)

    n_ops = sum(has_verifier_mean)
    print(f"Number of operations: {n_ops}\n")

    print(f"Number of operations requiring a verifier: {has_verifier_mean[0] / n_ops * 100}%\n")

    print(f"Number of operations requiring IRDL-C++ for args: {op_args_decl_mean[0] / n_ops * 100}%")

    print("-" * 60)
    print(25 * " " + "Operands")
    print("-" * 60)

    print(f"Number of operands:")
    print(f"\t0: {num_operands_mean[0] / sum(num_operands_mean) * 100}%")
    print(f"\t1: {num_operands_mean[1] / sum(num_operands_mean) * 100}%")
    print(f"\t2: {num_operands_mean[2] / sum(num_operands_mean) * 100}%")
    print(f"\t3+: {sum(num_operands_mean[3:]) / sum(num_operands_mean) * 100}%\n")

    print(f"Number of variadic operands:")
    print(f"\t0: {num_var_operands_mean[0] / sum(num_var_operands_mean) * 100}%")
    print(f"\t1+: {sum(num_var_operands_mean[1:]) / sum(num_var_operands_mean) * 100}%\n")

    print(f"Number of dialects with variadic operands: {len([None for _, val in num_var_operands_per_dialect.items() if sum(val[1:]) != 0]) / len(num_var_operands_per_dialect) * 100}%\n")

    print(f"Number of dialects with more than 25% of variadic operands: {len([None for _, val in num_var_operands_per_dialect.items() if 3 * sum(val[1:]) >= val[0]]) / len(num_var_operands_per_dialect) * 100}%\n")

    print("-" * 60)
    print(25 * " " + "Results")
    print("-" * 60)

    print(f"Number of results:")
    print(f"\t0: {num_results_mean[0] / sum(num_results_mean) * 100}%")
    print(f"\t1: {num_results_mean[1] / sum(num_results_mean) * 100}%")
    print(f"\t2: {num_results_mean[2] / sum(num_results_mean) * 100}%")

    print(f"Number of variadic results:")
    print(f"\t0: {num_var_results_mean[0] / sum(num_var_results_mean) * 100}%")
    print(f"\t1: {num_var_results_mean[1] / sum(num_var_results_mean) * 100}%")

    print(f"Number of dialects with variadic results: {len([None for _, val in num_var_results_per_dialect.items() if sum(val[1:]) != 0]) / len(num_var_results_per_dialect) * 100}%\n")

    print("-" * 60)
    print(22 * " " + "Operation Attributes")
    print("-" * 60)

    print(f"Number of attributes:")
    print(f"\t0: {num_attributes_mean[0] / sum(num_attributes_mean) * 100}%")
    print(f"\t1: {num_attributes_mean[1] / sum(num_attributes_mean) * 100}%")
    print(f"\t2: {num_attributes_mean[2] / sum(num_attributes_mean) * 100}%")
    print(f"\t2: {sum(num_attributes_mean[3:]) / sum(num_attributes_mean) * 100}%\n")

    print(f"Number of dialects with attributes: {len([None for _, val in num_attributes_per_dialect.items() if sum(val[1:]) != 0]) / len(num_attributes_per_dialect) * 100}%\n")

    print(f"Number of dialects with more than 25% of operations defining attributes: {len([None for _, val in num_attributes_per_dialect.items() if 3 * sum(val[1:]) >= val[0]]) / len(num_attributes_per_dialect) * 100}%\n")

    print("-" * 60)
    print(22 * " " + "Regions")
    print("-" * 60)

    print(f"Number of regions:")
    print(f"\t0: {num_regions_mean[0] / sum(num_regions_mean) * 100}%")
    print(f"\t1: {num_regions_mean[1] / sum(num_regions_mean) * 100}%")
    print(f"\t2: {num_regions_mean[2] / sum(num_regions_mean) * 100}%\n")

    print(f"Number of dialects with regions: {len([None for _, val in num_regions_per_dialect.items() if sum(val[1:]) != 0]) / len(num_regions_per_dialect) * 100}%\n")

    print("-" * 60)
    print(22 * " " + "Attributes and Types")
    print("-" * 60)

    n_types = sum([sum(val) for val in type_verifier.values()])
    n_attrs = sum([sum(val) for val in attr_verifier.values()])

    print(f"Number of types defined: {n_types}\n")
    print(f"Number of attributes defined: {n_attrs}\n")

    n_parameters = sum([sum(val) for val in type_param_groups.values()]) + sum([sum(val) for val in attr_param_groups.values()])
    n_non_dialect_specific_parameters = sum(type_param_groups['llvm']) + sum(type_param_groups['affine']) + sum(attr_param_groups['affine'])
    print(f"Proportion of types and attributes parameters not in IRDL: { n_non_dialect_specific_parameters / n_parameters * 100}%\n")

    print(f"Number of types that use a non-IRDL parameter: {sum([val[0] for val in type_params_decl.values()]) / sum([sum(val) for val in type_params_decl.values()]) * 100}%\n")

    print(f"Number of attributes that use a non-IRDL parameter: {sum([val[0] for val in attr_params_decl.values()]) / sum([sum(val) for val in attr_params_decl.values()]) * 100}%\n")

    print(f"Number of types defining a verifier: {sum([val[1] for val in type_verifier.values()]) / n_types * 100}%\n")

    print(f"Number of attributes defining a verifier: {sum([val[1] for val in attr_verifier.values()]) / n_attrs * 100}%\n")


def generate_evolution_plot(output_dir):
    # Month	Dialects Operations Types Attributes
    data = [
        ["11/2021", 31, 942, 62, 32],
        ["10/2021", 30, 927, 62, 32],
        ["09/2021", 30, 923, 62, 31],
        ["08/2021", 30, 940, 62, 31],
        ["07/2021", 30, 940, 62, 31],
        ["06/2021", 29, 909, 61, 30],
        ["05/2021", 29, 866, 60, 30],
        ["04/2021", 29, 852, 60, 29],
        ["03/2021", 28, 825, 59, 26],
        ["02/2021", 28, 785, 58, 25],
        ["01/2021", 27, 746, 60, 24],
        ["12/2020", 22, 725, 61, 24],
        ["11/2020", 21, 630, 59, 24],
        ["10/2020", 21, 616, 58, 24],
        ["09/2020", 20, 592, 56, 24],
        ["08/2020", 17, 533, 31, 24],
        ["07/2020", 17, 510, 31, 24],
        ["06/2020", 17, 498, 30, 25],
        ["05/2020", 17, 442, 28, 25],
        ["04/2020", 18, 444, 28, 23]
    ]

    fig, ax = plt.subplots()
    dates = list(map(lambda x: x[0], data))
    operations = list(map(lambda x: x[2], data))
    dates = [dt.datetime.strptime(d, '%m/%Y').date() for d in dates]
    ax.plot(dates, operations, marker='.', c='#a6cee3', mfc='#1f78b4', mec='#1f78b4')

    ax.set(xlabel='', ylabel='Operations')
    plt.xticks(rotation=60)
    ax.set_ylim(200, 1000)
    ax.grid(axis='y')

    x1, y1 = [dates[0], dates[-1]], [400, 400]
    plt.arrow(x1[1], y1[0], 585, 1, color="black", width=5, zorder=100, length_includes_head=True, head_width=25,
              head_length=5)
    plt.text(dates[0] + (dates[-1] - dates[0]) / 2, 300, "20 months", ha="center", weight="bold", color="black")

    nd = dt.datetime.strptime('12/2021', '%m/%Y').date()
    x1, y1 = [nd, nd], [444, 942]
    plt.arrow(x1[0], y1[0], 0, y1[1] - y1[0], color="black", width=2, head_width=10, zorder=100,
              length_includes_head=True)

    nd = dt.datetime.strptime('01/2022', '%m/%Y').date()
    plt.text(nd, 660, "2.1×\n#ops", ha="left", weight="bold", color="black")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%m/%y')
    ax.xaxis.set_major_formatter(myFmt)

    fig = mpl.pyplot.gcf()
    fig.set_size_inches(5, 3)

    plt.tight_layout()
    plt.savefig(output_dir + "/evolution.pdf", metadata={'CreationDate': None})


arg_parser = argparse.ArgumentParser(description='Generate plots used in the IRDL paper')
arg_parser.add_argument("--input-file",
                        type=str,
                        nargs="?",
                        required=True,
                        help="Path to the JSON file containing MLIR dialects data")
arg_parser.add_argument("--output-dir",
                        type=str,
                        nargs="?",
                        required=True,
                        help="Path to the root directory containing the plots")

if __name__ == "__main__":
    args = arg_parser.parse_args()
    f = open(args.input_file, "r")
    stats = get_stats_from_json(f.read())
    gather_all_plots_data(stats)
    generate_evolution_plot(args.output_dir)
    print_data_used_in_text()

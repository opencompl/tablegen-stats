from __future__ import annotations

from extraction import *
from analyze_tablegen.irdl import *

LLVM_ROOT = "../../../../llvm-project"


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


def get_traits_culprits(stats: Stats) -> Dict[Trait, int]:
    culprits = dict()
    for op in stats.ops:
        for trait in op.traits:
            if not trait.is_declarative():
                culprits.setdefault(trait, 0)
                culprits[trait] += 1
    return culprits


def get_interfaces_culprits(stats: Stats) -> Dict[str, int]:
    culprits = dict()
    for op in stats.ops:
        for interface in op.interfaces:
            culprits.setdefault(interface, 0)
            culprits[interface] += 1
    return culprits


def get_type_param_culprits(stats: Stats) -> Dict[str, int]:
    culprits = dict()
    for typ in stats.types:
        for param in typ.parameters:
            if not param.is_declarative():
                culprits.setdefault(param.cppType, 0)
                culprits[param.cppType] += 1
    return culprits


def get_attr_param_culprits(stats: Stats) -> Dict[AttrOrTypeParameter, int]:
    culprits = dict()
    for attr in stats.attrs:
        for param in attr.parameters:
            if not param.is_declarative():
                culprits.setdefault(param.cppType, 0)
                culprits[param.cppType] += 1
    return culprits


def create_type_parameters_decl_plot(stats: Stats):
    type_res = get_global_type_distribution(
        stats,
        lambda typ: int(all([p.is_declarative()
                             for p in typ.parameters])))[1]
    attr_res = get_global_attr_distribution(
        stats,
        lambda attr: int(all([p.is_declarative()
                              for p in attr.parameters])))[1]
    # noinspection PyTypeChecker
    type_res = dict(filter(lambda x: sum(x[1]) != 0, type_res.items()))
    # noinspection PyTypeChecker
    attr_res = dict(filter(lambda x: sum(x[1]) != 0, attr_res.items()))
    print(f"type_params_decl = {type_res}")
    print(f"attr_params_decl = {attr_res}")


def create_type_verifiers_plot(stats: Stats):
    type_res = get_global_type_distribution(
        stats, lambda typ: int(not typ.hasVerifier))[1]
    attr_res = get_global_attr_distribution(
        stats, lambda attr: int(not attr.hasVerifier))[1]
    # noinspection PyTypeChecker
    type_res = dict(filter(lambda x: sum(x[1]) != 0, type_res.items()))
    # noinspection PyTypeChecker
    attr_res = dict(filter(lambda x: sum(x[1]) != 0, attr_res.items()))
    print(f"type_verifier = {type_res}")
    print(f"attr_verifier = {attr_res}")


def create_type_parameters_type_plot(stats: Stats):
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

    print(f"type_param_groups = {distr_to_group(distr_typ)}")
    print(f"attr_param_groups = {distr_to_group(distr_attr)}")


def create_op_stats_plots(stats: Stats):
    num_operands = get_global_op_distribution(stats, lambda x: len(x.operands))
    print(f"num_operands_mean = {num_operands[0]}")
    print(f"num_operands_per_dialect = {num_operands[1]}")
    num_var_operands = get_global_op_distribution(
        stats, lambda x: sum([
            isinstance(operand.constraint, VariadicConstraint)
            for operand in x.operands
        ]))
    print(f"num_var_operands_mean = {num_var_operands[0]}")
    print(f"num_var_operands_per_dialect = {num_var_operands[1]}")
    num_results = get_global_op_distribution(stats, lambda x: len(x.results))
    print(f"num_results_mean = {num_results[0]}")
    print(f"num_results_per_dialect = {num_results[1]}")
    num_var_results = get_global_op_distribution(
        stats, lambda x: sum([
            isinstance(result.constraint, VariadicConstraint)
            for result in x.results
        ]))
    print(f"num_var_results_mean = {num_var_results[0]}")
    print(f"num_var_results_per_dialect = {num_var_results[1]}")
    num_attributes = get_global_op_distribution(stats,
                                                lambda x: len(x.attributes))
    print(f"num_attributes_mean = {num_attributes[0]}")
    print(f"num_attributes_per_dialect = {num_attributes[1]}")
    num_regions = get_global_op_distribution(stats, lambda x: len(x.regions))
    print(f"num_regions_mean = {num_regions[0]}")
    print(f"num_regions_per_dialect = {num_regions[1]}")
    has_verifiers = get_global_op_distribution(stats,
                                               lambda x: int(x.hasVerifier))
    print(f"has_verifier_mean = {has_verifiers[0]}")
    print(f"has_verifier_per_dialect = {has_verifiers[1]}")
    op_args_decl = get_global_op_distribution(
        stats, lambda x: int(x.is_operands_results_attrs_declarative()))
    print(f"op_args_decl_mean = {op_args_decl[0]}")
    print(f"op_args_decl_per_dialect = {op_args_decl[1]}")
    op_non_decl_constraints = get_constraints_culprits(stats)
    print(f"op_non_decl_constraints = {op_non_decl_constraints}")


def __main__():
    # stats = get_stat_from_files(llvm_root)

    f = open("tablegen_data.json", "r")
    stats = get_stats_from_json(f.read())

    print(stats.types)
    print(stats.attrs)

    print("-" * 80)
    print("Culprits:")
    print("-" * 80)

    print("Constraints:")
    constraints_culprits = list(get_constraints_culprits(stats).items())
    list.sort(constraints_culprits, key=lambda x: x[1], reverse=True)
    print(constraints_culprits)

    print("Traits:")
    traits_culprits = list(get_traits_culprits(stats).items())
    list.sort(traits_culprits, key=lambda x: x[1], reverse=True)
    print(traits_culprits)

    print("Interfaces:")
    traits_culprits = list(get_interfaces_culprits(stats).items())
    list.sort(traits_culprits, key=lambda x: x[1], reverse=True)
    print(traits_culprits)

    print("Type params:")
    type_culprits = list(get_type_param_culprits(stats).items())
    list.sort(type_culprits, key=lambda x: x[1], reverse=True)
    print(type_culprits)

    print("Attr params:")
    attr_culprits = list(get_attr_param_culprits(stats).items())
    list.sort(attr_culprits, key=lambda x: x[1], reverse=True)
    print(attr_culprits)

    print("-" * 80)
    print("Some general stats:")
    print("-" * 80)

    print("Number of operations defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.ops), x.numOperations)))
    print("total:", (
        len(stats.ops),
        sum([dialect.numOperations for dialect in stats.dialects.values()]),
    ))

    print("Number of types defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.types), x.numTypes)))
    print("total:", (
        len(stats.types),
        sum([dialect.numTypes for dialect in stats.dialects.values()]),
    ))

    print("Number of attributes defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.attrs), x.numAttributes)))
    print("total:", (
        len(stats.attrs),
        sum([dialect.numAttributes for dialect in stats.dialects.values()]),
    ))

    # create_type_parameters_type_plot(stats)
    # create_type_parameters_decl_plot(stats)
    # create_type_verifiers_plot(stats)
    create_op_stats_plots(stats)


if __name__ == "__main__":
    #res = get_files_contents_as_json()
    #f = open("tablegen_data.json", "w")
    #f.write(res)
    #f.close()

    __main__()

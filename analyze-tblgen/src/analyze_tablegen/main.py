from __future__ import annotations

from analyze_tablegen.extraction import *
from analyze_tablegen.irdl import *

LLVM_ROOT = "../../../../llvm-project"


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


def get_param_culprits(stats: Stats) -> Dict[str, int]:
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


def get_type_param_culprits(stats: Stats) -> Dict[AttrOrTypeParameter, int]:
    culprits = dict()
    for typ in stats.types:
        for param in typ.parameters:
            if not param.is_declarative():
                culprits.setdefault(param.cppType, 0)
                culprits[param.cppType] += 1
    return culprits


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


if __name__ == "__main__":
    __main__()

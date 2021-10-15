from __future__ import annotations
import json
import os
import subprocess
from irdl import *

verifiers_allow_len_equality = False
llvm_root = "../../llvm-project"


# OPENMP and OPENACC are not included since they rely on generated tablegen
# files
# DLTI is removed because it contains no operations
def get_tablegen_op_file_list():
    res_files = []
    for root, dirs, files in os.walk(llvm_root + "/mlir/include"):
        for file in files:
            if file.endswith(".td"):
                res_files.append(os.path.join(root, file))
    return res_files


def get_stat_from_file(file) -> Optional[Stats]:
    root, file = os.path.split(file)
    res = subprocess.run(["../build/bin/tblgen-extract", os.path.join(root, file), f"--I={llvm_root}/mlir/include",
                          f"--I={root}"], capture_output=True)
    if res.returncode != 0:
        return None
    print("../build/bin/tblgen-extract", os.path.join(root, file), f"--I={llvm_root}/mlir/include",
          f"--I={root}")
    ops = json.loads(res.stdout)
    return Stats.from_json(ops)


def add_cpp_types(stats: Stats):
    # linalg
    stats.add_type(Type("range", "linalg", False, [], [], []))

    # gpu
    stats.add_type(Type("async.token", "gpu", False, [], [], []))
    # verifyCompatibleShape
    stats.add_type(Type("mma_matrix", "gpu", True, [AttrOrTypeParameter("shape_x", "int64_t"),
                                                    AttrOrTypeParameter("shape_y", "int64_t"),
                                                    AttrOrTypeParameter("elementType", "Type"),
                                                    AttrOrTypeParameter("operand", "StringAttr")], [], []))
    # spv
    stats.add_type(Type("array", "spv", False, [AttrOrTypeParameter("elementType", "Type"),
                                                AttrOrTypeParameter("elementCount", "unsigned"),
                                                AttrOrTypeParameter("stride", "unsigned")], [], []))
    stats.add_type(Type("coopmatrix", "spv", False, [AttrOrTypeParameter("elementType", "Type"),
                                                     AttrOrTypeParameter("rows", "unsigned"),
                                                     AttrOrTypeParameter("columns", "unsigned"),
                                                     AttrOrTypeParameter("scope", "Scope")], [], []))
    stats.add_type(Type("image", "spv", False, [AttrOrTypeParameter("elementType", "Type"),
                                                AttrOrTypeParameter("dim", "Dim"),
                                                AttrOrTypeParameter("depthInfo", "ImageDepthInfo"),
                                                AttrOrTypeParameter("arrayedInfo", "ImageArrayedInfo"),
                                                AttrOrTypeParameter("samplingInfo", "ImageSamplingInfo"),
                                                AttrOrTypeParameter("samplerUseInfo", "ImageSamplerUseInfo"),
                                                AttrOrTypeParameter("format", "ImageFormat")], [], []))
    stats.add_type(Type("ptr", "spv", False, [AttrOrTypeParameter("pointeeType", "Type"),
                                              AttrOrTypeParameter("storageClass", "StorageClass")], [], []))
    stats.add_type(Type("rtarray", "spv", False, [AttrOrTypeParameter("elementType", "Type"),
                                                  AttrOrTypeParameter("stride", "int")], [], []))
    stats.add_type(Type("sampled_image", "spv", False, [AttrOrTypeParameter("imageType", "Type")], [], []))
    stats.add_type(Type("struct", "spv", False, [AttrOrTypeParameter("memberTypes", "ArrayRef<Type>"),
                                                 AttrOrTypeParameter("offsetInfo", "ArrayRef<OffsetInfo>"),
                                                 AttrOrTypeParameter("memberDecorations",
                                                                     "ArrayRef<MemberDecorationInfo>")], [],
                        []))
    stats.add_type(Type("matrix", "spv", False, [AttrOrTypeParameter("columnType", "Type"),
                                                 AttrOrTypeParameter("columnCount", "uint32_t")], [], []))

    # llvm
    stats.add_type(Type("void", "llvm", False, [], [], []))
    stats.add_type(Type("ppc_fp128", "llvm", False, [], [], []))
    stats.add_type(Type("x86mmx", "llvm", False, [], [], []))
    stats.add_type(Type("token", "llvm", False, [], [], []))
    stats.add_type(Type("label", "llvm", False, [], [], []))
    stats.add_type(Type("metadata", "llvm", False, [], [], []))
    stats.add_type(Type("func", "llvm", False, [AttrOrTypeParameter("result", "Type"),
                                                AttrOrTypeParameter("arguments", "ArrayRef<Type>"),
                                                AttrOrTypeParameter("isVarArg", "bool")], [], []))
    stats.add_type(Type("ptr", "llvm", False, [AttrOrTypeParameter("pointee", "Type"),
                                               AttrOrTypeParameter("addressSpace", "unsigned")], [],
                        ["DataLayoutTypeInterface::Trait"]))
    # Check that a value is strictly positive
    stats.add_type(Type("fixed_vec", "llvm", True, [AttrOrTypeParameter("elementType", "Type"),
                                                    AttrOrTypeParameter("numElements", "unsigned")], [], []))
    # Check that a value is strictly positive
    stats.add_type(Type("scalable_vec", "llvm", True, [AttrOrTypeParameter("elementType", "Type"),
                                                       AttrOrTypeParameter("numElements", "unsigned")], [],
                        []))
    stats.add_type(Type("array", "llvm", False, [AttrOrTypeParameter("elementType", "Type"),
                                                 AttrOrTypeParameter("numElements", "unsigned")], [], []))
    # Complex underlying type that requires non-trivial verifier
    stats.add_type(Type("struct", "llvm", True, [AttrOrTypeParameter("arg", "LLVMStruct")], [], []))

    # shape
    stats.add_type(Type("shape", "shape", False, [], [], []))
    stats.add_type(Type("size", "shape", False, [], [], []))
    stats.add_type(Type("value_shape", "shape", False, [], [], []))
    stats.add_type(Type("witness", "shape", False, [], [], []))

    # quant
    # Complex verifier
    stats.add_type(Type("any", "quant", False, [AttrOrTypeParameter("flags", "unsigned"),
                                                AttrOrTypeParameter("storageType", "Type"),
                                                AttrOrTypeParameter("expressedType", "Type"),
                                                AttrOrTypeParameter("storageTypeMin", "int64_t"),
                                                AttrOrTypeParameter("storageTypeMax", "int64_t")], [], []))
    # Complex verifier
    stats.add_type(Type("uniform", "quant", False, [AttrOrTypeParameter("flags", "unsigned"),
                                                    AttrOrTypeParameter("storageType", "Type"),
                                                    AttrOrTypeParameter("expressedType", "Type"),
                                                    AttrOrTypeParameter("scale", "double"),
                                                    AttrOrTypeParameter("zeroPoint", "int64_t"),
                                                    AttrOrTypeParameter("storageTypeMin", "int64_t"),
                                                    AttrOrTypeParameter("storageTypeMax", "int64_t")], [],
                        []))
    # Complex verifier
    stats.add_type(Type("uniform_per_axis", "quant", False, [AttrOrTypeParameter("flags", "unsigned"),
                                                             AttrOrTypeParameter("storageType", "Type"),
                                                             AttrOrTypeParameter("expressedType", "Type"),
                                                             AttrOrTypeParameter("scales",
                                                                                 "ArrayRef<double>"),
                                                             AttrOrTypeParameter("zeroPoints",
                                                                                 "ArrayRef<int64_t>"),
                                                             AttrOrTypeParameter("quantizedDimension",
                                                                                 "int64_t"),
                                                             AttrOrTypeParameter("storageTypeMin", "int64_t"),
                                                             AttrOrTypeParameter("storageTypeMax",
                                                                                 "int64_t")], [], []))
    # Less or equal comparison
    stats.add_type(Type("calibrated", "quant", False, [AttrOrTypeParameter("expressedType", "Type"),
                                                       AttrOrTypeParameter("min", "double"),
                                                       AttrOrTypeParameter("max", "double")], [], []))


def add_cpp_attributes(stats: Stats):
    # spv
    stats.add_attr(Attr("interface_var_abi", "spv", False, [AttrOrTypeParameter("descriptorSet", "uint32_t"),
                                                            AttrOrTypeParameter("binding", "uint32_t"),
                                                            AttrOrTypeParameter("storageClass",
                                                                                "Optional<StorageClass>")],
                        [], []))
    stats.add_attr(Attr("ver_cap_ext", "spv", False, [AttrOrTypeParameter("version", "Version"),
                                                      AttrOrTypeParameter("capabilities",
                                                                          "ArrayRef<Capability>"),
                                                      AttrOrTypeParameter("extensions",
                                                                          "ArrayRef<Extension>")], [], []))
    stats.add_attr(Attr("target_env", "spv", False, [AttrOrTypeParameter("triple", "VerCapExtAttr"),
                                                     AttrOrTypeParameter("vendorID", "Vendor"),
                                                     AttrOrTypeParameter("deviceType", "DeviceType"),
                                                     AttrOrTypeParameter("deviceId", "uint32_t"),
                                                     AttrOrTypeParameter("limits", "DictionaryAttr")], [],
                        []))

    # vector
    stats.add_attr(
        Attr("combining_kind", "vector", False, [AttrOrTypeParameter("kind", "CombiningKind")], [], []))


def remove_unnecessary_verifiers(stats: Stats):
    # Types:
    stats.dialects["builtin"].types["Builtin_Complex"].hasVerifier = False
    stats.dialects["builtin"].types["Builtin_UnrankedMemRef"].hasVerifier = False
    stats.dialects["builtin"].types["Builtin_UnrankedTensor"].hasVerifier = False

    # Attributes

    # Linalg has no unnecessary verifiers
    # gpu
    stats.dialects["gpu"].ops["gpu.block_dim"].hasVerifier = False
    stats.dialects["gpu"].ops["gpu.block_id"].hasVerifier = False
    stats.dialects["gpu"].ops["gpu.grid_dim"].hasVerifier = False
    stats.dialects["gpu"].ops["gpu.thread_id"].hasVerifier = False
    stats.dialects["gpu"].ops["gpu.shuffle"].hasVerifier = False
    # amx
    # x86vector
    # tensor
    if verifiers_allow_len_equality:
        stats.dialects["tensor"].ops["tensor.extract"].hasVerifier = False
        stats.dialects["tensor"].ops["tensor.insert"].hasVerifier = False
    # affine
    # emitc
    stats.dialects["emitc"].ops["emitc.apply"].hasVerifier = False


def get_stat_from_files():
    stats = Stats()
    for file in get_tablegen_op_file_list():
        file_stats = get_stat_from_file(file)
        if file_stats is not None:
            stats.add_stats(file_stats)

    add_cpp_types(stats)
    add_cpp_attributes(stats)
    remove_unnecessary_verifiers(stats)

    # res = subprocess.run(["../build/bin/dialect-extract"], capture_output=True)
    # if res.returncode != 0:
    #     return None
    # data = json.loads(res.stderr)
    # for json_dialect in data:
    #     dialect_stats = stats.dialects[json_dialect["name"]]
    #     dialect_stats.numOperations = json_dialect["numOperations"]
    #     dialect_stats.numTypes = json_dialect["numTypes"]
    #     dialect_stats.numAttributes = json_dialect["numAttributes"]

    return stats


def get_op_list_distribution(ops: List[Op], f: Callable[[Op], int], maxi: Optional[int] = None) -> List[float]:
    res = [f(op) for op in ops]
    if maxi is None:
        maxi = max(res)
    return [res.count(i) for i in range(maxi + 1)]


def get_global_op_distribution(stats: Stats, f: Callable[[Op], int]):
    global_res = get_op_list_distribution(stats.ops, f)
    maxi = len(global_res) - 1
    per_dialect_res = dict()
    for dialect_name, dialect in stats.dialects.items():
        per_dialect_res[dialect_name] = get_op_list_distribution(list(dialect.ops.values()), f, maxi=maxi)
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
        per_dialect_res[dialect_name] = get_attr_list_distribution(list(dialect.attrs.values()), f, maxi=maxi)
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
        per_dialect_res[dialect_name] = get_type_list_distribution(list(dialect.types.values()), f, maxi=maxi)
    return global_res, per_dialect_res


T = TypeVar("T")


def get_dialect_values(stats: Stats, f: Callable[[Dialect], T]) -> Dict[str, T]:
    return {key: f(value) for key, value in stats.dialects.items()}


def add_non_declarative_constraint(constraint: Constraint, d: Dict[Constraint, int]):
    if isinstance(constraint, PredicateConstraint):
        if not constraint.is_declarative():
            if constraint in d:
                d[constraint] += 1
            else:
                d[constraint] = 1
    for sub_constraint in constraint.get_sub_constraints():
        add_non_declarative_constraint(sub_constraint, d)


def get_constraints_culprits(stats: Stats) -> Dict[Constraint, int]:
    culprits = dict()
    for op in stats.ops:
        for operand in op.operands:
            constraint = operand.constraint
            add_non_declarative_constraint(constraint, culprits)
        for result in op.results:
            constraint = result.constraint
            add_non_declarative_constraint(constraint, culprits)
        for attr in op.attributes.values():
            add_non_declarative_constraint(attr, culprits)
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


def create_type_attr_evolution_per_dialect_decl_plot(stats: Stats):
    default_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=False, enums=False)))[1]
    builtins_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=False)))[1]
    enums_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]

    default_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=False, enums=False)))[1]
    builtins_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=False)))[1]
    enums_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]

    attributes = dict()
    types = dict()
    for key in default_attr:
        attr_sum = default_attr[key][1] + default_attr[key][0]
        if attr_sum != 0:
            res = (default_attr[key][1], builtins_attr[key][1] - default_attr[key][1],
                   enums_attr[key][1] - builtins_attr[key][1], attr_sum - enums_attr[key][1])
            attributes[key] = (
                res[0] / attr_sum * 100, res[1] / attr_sum * 100, res[2] / attr_sum * 100, res[3] / attr_sum * 100)
        type_sum = default_type[key][1] + default_type[key][0]
        if type_sum != 0:
            res = (default_type[key][1], builtins_type[key][1] - default_type[key][1],
                   enums_type[key][1] - builtins_type[key][1], type_sum - enums_type[key][1])
            types[key] = (
                res[0] / type_sum * 100, res[1] / type_sum * 100, res[2] / type_sum * 100, res[3] / type_sum * 100)

    print(attributes)
    print(types)


def create_type_attr_evolution_decl_plot(stats: Stats):
    default_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=False, enums=False)))[0]
    builtins_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=False)))[0]
    enums_attr = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[0]

    default_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=False, enums=False)))[0]
    builtins_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=False)))[0]
    enums_type = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[0]

    def mp(v):
        return (v[1] / (v[0] + v[1])) * 100

    attrs = (mp(default_attr), mp(builtins_attr), mp(enums_attr))
    types = (mp(default_type), mp(builtins_type), mp(enums_type))
    print(attrs)
    print(types)


def create_dialects_decl_plot(stats: Stats):
    types = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]
    types = {key: value[1] / sum(value) * 100 for key, value in types.items() if sum(value) != 0}
    attrs = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]
    attrs = {key: value[1] / sum(value) * 100 for key, value in attrs.items() if sum(value) != 0}

    op_operands = get_global_op_distribution(stats, lambda x: 1 if x.is_operands_results_attrs_declarative() else 0)[1]
    op_operands = {key: value[1] / sum(value) * 100 for key, value in op_operands.items() if sum(value) != 0}
    op_full = get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(check_traits=True,
                                                                                check_interfaces=False) else 0)[1]
    op_full = {key: value[1] / sum(value) * 100 for key, value in op_full.items() if sum(value) != 0}

    print(types)
    print(attrs)
    print(op_operands)
    print(op_full)


def create_dialects_decl_plot2(stats: Stats):
    types = get_global_type_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]
    types = {key: value[1] / sum(value) * 100 for key, value in types.items() if sum(value) != 0}
    attrs = get_global_attr_distribution(stats, lambda x: int(x.is_declarative(builtins=True, enums=True)))[1]
    attrs = {key: value[1] / sum(value) * 100 for key, value in attrs.items() if sum(value) != 0}

    op_operands = get_global_op_distribution(stats, lambda x: 1 if x.is_operands_results_attrs_declarative() else 0)[1]
    op_full = get_global_op_distribution(stats, lambda x: 1 if x.is_declarative(check_traits=True,
                                                                                check_interfaces=False) else 0)[1]
    ops = {key: (op_full[key][1], value[1] - op_full[key][1], sum(value) - value[1]) for key, value in
           op_operands.items() if sum(value) != 0}

    print(types)
    print(attrs)
    print(ops)


def create_type_parameters_type_plot(stats: Stats):
    distr = dict()
    for typ in stats.types:
        for param in typ.parameters:
            distr.setdefault(param.get_group(), 0)
            distr[param.get_group()] += 1
    for attr in stats.attrs:
        for param in attr.parameters:
            distr.setdefault(param.get_group(), 0)
            distr[param.get_group()] += 1
    print(distr)


def __main__():
    stats = get_stat_from_files()

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
    print("total:", (len(stats.ops), sum([dialect.numOperations for dialect in stats.dialects.values()]),))

    print("Number of types defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.types), x.numTypes)))
    print("total:", (len(stats.types), sum([dialect.numTypes for dialect in stats.dialects.values()]),))

    print("Number of attributes defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.attrs), x.numAttributes)))
    print("total:", (len(stats.attrs), sum([dialect.numAttributes for dialect in stats.dialects.values()]),))

    # create_type_attr_evolution_per_dialect_decl_plot(stats)
    # create_type_attr_evolution_decl_plot(stats)
    # create_dialects_decl_plot2(stats)
    # create_type_parameters_type_plot(stats)


if __name__ == "__main__":
    __main__()

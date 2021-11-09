from __future__ import annotations
import json
import os
import subprocess
from analyze_tablegen.irdl import *

verifiers_allow_len_equality = False
llvm_root = "../../../../llvm-project"


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


def get_file_contents(file) -> Optional[str]:
    root, file = os.path.split(file)
    res = subprocess.run(["../../../build/bin/tblgen-extract", os.path.join(root, file), f"--I={llvm_root}/mlir/include",
                          f"--I={root}"], capture_output=True)
    if res.returncode != 0:
        return None
    print("../build/bin/tblgen-extract", os.path.join(root, file), f"--I={llvm_root}/mlir/include",
          f"--I={root}")
    return json.loads(res.stdout)


def get_stat_from_file(file) -> Optional[Stats]:
    contents = get_file_contents(file)
    if contents is None:
        return None
    return Stats.from_json(contents)


def add_cpp_types(stats: Stats):
    # linalg
    stats.add_type(Type("range", "linalg", "UNKNOWN", False, [], [], []))

    # gpu
    stats.add_type(Type("async.token", "gpu", "UNKNOWN", False, [], [], []))
    # verifyCompatibleShape
    stats.add_type(Type("mma_matrix", "gpu", "UNKNOWN", True, [AttrOrTypeParameter("shape_x", "int64_t"),
                                                    AttrOrTypeParameter("shape_y", "int64_t"),
                                                    AttrOrTypeParameter("elementType", "Type"),
                                                    AttrOrTypeParameter("operand", "StringAttr")], [], []))
    # spv
    stats.add_type(Type("array", "spv", "UNKNOWN", False, [AttrOrTypeParameter("elementType", "Type"),
                                                AttrOrTypeParameter("elementCount", "unsigned"),
                                                AttrOrTypeParameter("stride", "unsigned")], [], []))
    stats.add_type(Type("coopmatrix", "spv", "UNKNOWN", False, [AttrOrTypeParameter("elementType", "Type"),
                                                     AttrOrTypeParameter("rows", "unsigned"),
                                                     AttrOrTypeParameter("columns", "unsigned"),
                                                     AttrOrTypeParameter("scope", "Scope")], [], []))
    stats.add_type(Type("image", "spv", "UNKNOWN", False, [AttrOrTypeParameter("elementType", "Type"),
                                                AttrOrTypeParameter("dim", "Dim"),
                                                AttrOrTypeParameter("depthInfo", "ImageDepthInfo"),
                                                AttrOrTypeParameter("arrayedInfo", "ImageArrayedInfo"),
                                                AttrOrTypeParameter("samplingInfo", "ImageSamplingInfo"),
                                                AttrOrTypeParameter("samplerUseInfo", "ImageSamplerUseInfo"),
                                                AttrOrTypeParameter("format", "ImageFormat")], [], []))
    stats.add_type(Type("ptr", "spv", "UNKNOWN", False, [AttrOrTypeParameter("pointeeType", "Type"),
                                              AttrOrTypeParameter("storageClass", "StorageClass")], [], []))
    stats.add_type(Type("rtarray", "spv", "UNKNOWN", False, [AttrOrTypeParameter("elementType", "Type"),
                                                  AttrOrTypeParameter("stride", "int")], [], []))
    stats.add_type(Type("sampled_image", "spv", "UNKNOWN", False, [AttrOrTypeParameter("imageType", "Type")], [], []))
    stats.add_type(Type("struct", "spv", "UNKNOWN", False, [AttrOrTypeParameter("memberTypes", "ArrayRef<Type>"),
                                                 AttrOrTypeParameter("offsetInfo", "ArrayRef<OffsetInfo>"),
                                                 AttrOrTypeParameter("memberDecorations",
                                                                     "ArrayRef<MemberDecorationInfo>")], [],
                        []))
    stats.add_type(Type("matrix", "spv", "UNKNOWN", False, [AttrOrTypeParameter("columnType", "Type"),
                                                 AttrOrTypeParameter("columnCount", "uint32_t")], [], []))

    # llvm
    stats.add_type(Type("void", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("ppc_fp128", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("x86mmx", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("token", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("label", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("metadata", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("func", "llvm", "UNKNOWN", False, [AttrOrTypeParameter("result", "Type"),
                                                AttrOrTypeParameter("arguments", "ArrayRef<Type>"),
                                                AttrOrTypeParameter("isVarArg", "bool")], [], []))
    stats.add_type(Type("ptr", "llvm", "UNKNOWN", False, [AttrOrTypeParameter("pointee", "Type"),
                                               AttrOrTypeParameter("addressSpace", "unsigned")], [],
                        ["DataLayoutTypeInterface::Trait"]))
    # Check that a value is strictly positive
    stats.add_type(Type("fixed_vec", "llvm", "UNKNOWN", True, [AttrOrTypeParameter("elementType", "Type"),
                                                    AttrOrTypeParameter("numElements", "unsigned")], [], []))
    # Check that a value is strictly positive
    stats.add_type(Type("scalable_vec", "llvm", "UNKNOWN", True, [AttrOrTypeParameter("elementType", "Type"),
                                                       AttrOrTypeParameter("numElements", "unsigned")], [],
                        []))
    stats.add_type(Type("array", "llvm", "UNKNOWN", False, [AttrOrTypeParameter("elementType", "Type"),
                                                 AttrOrTypeParameter("numElements", "unsigned")], [], []))
    # Complex underlying type that requires non-trivial verifier
    stats.add_type(Type("struct", "llvm", "UNKNOWN", True, [AttrOrTypeParameter("arg", "LLVMStruct")], [], []))

    # shape
    stats.add_type(Type("shape", "shape", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("size", "shape", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("value_shape", "shape", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("witness", "shape", "UNKNOWN", False, [], [], []))

    # quant
    # Complex verifier
    stats.add_type(Type("any", "quant", "UNKNOWN", False, [AttrOrTypeParameter("flags", "unsigned"),
                                                AttrOrTypeParameter("storageType", "Type"),
                                                AttrOrTypeParameter("expressedType", "Type"),
                                                AttrOrTypeParameter("storageTypeMin", "int64_t"),
                                                AttrOrTypeParameter("storageTypeMax", "int64_t")], [], []))
    # Complex verifier
    stats.add_type(Type("uniform", "quant", "UNKNOWN", False, [AttrOrTypeParameter("flags", "unsigned"),
                                                    AttrOrTypeParameter("storageType", "Type"),
                                                    AttrOrTypeParameter("expressedType", "Type"),
                                                    AttrOrTypeParameter("scale", "double"),
                                                    AttrOrTypeParameter("zeroPoint", "int64_t"),
                                                    AttrOrTypeParameter("storageTypeMin", "int64_t"),
                                                    AttrOrTypeParameter("storageTypeMax", "int64_t")], [],
                        []))
    # Complex verifier
    stats.add_type(Type("uniform_per_axis", "quant", "UNKNOWN", False, [AttrOrTypeParameter("flags", "unsigned"),
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
    stats.add_type(Type("calibrated", "quant", "UNKNOWN", False, [AttrOrTypeParameter("expressedType", "Type"),
                                                       AttrOrTypeParameter("min", "double"),
                                                       AttrOrTypeParameter("max", "double")], [], []))


def add_cpp_attributes(stats: Stats):
    # spv
    stats.add_attr(Attr("interface_var_abi", "spv", "UNKNOWN", False, [AttrOrTypeParameter("descriptorSet", "uint32_t"),
                                                            AttrOrTypeParameter("binding", "uint32_t"),
                                                            AttrOrTypeParameter("storageClass",
                                                                                "Optional<StorageClass>")],
                        [], []))
    stats.add_attr(Attr("ver_cap_ext", "spv", "UNKNOWN", False, [AttrOrTypeParameter("version", "Version"),
                                                      AttrOrTypeParameter("capabilities",
                                                                          "ArrayRef<Capability>"),
                                                      AttrOrTypeParameter("extensions",
                                                                          "ArrayRef<Extension>")], [], []))
    stats.add_attr(Attr("target_env", "spv", "UNKNOWN", False, [AttrOrTypeParameter("triple", "VerCapExtAttr"),
                                                     AttrOrTypeParameter("vendorID", "Vendor"),
                                                     AttrOrTypeParameter("deviceType", "DeviceType"),
                                                     AttrOrTypeParameter("deviceId", "uint32_t"),
                                                     AttrOrTypeParameter("limits", "DictionaryAttr")], [],
                        []))

    # vector
    stats.add_attr(
        Attr("combining_kind", "vector", "UNKNOWN", False, [AttrOrTypeParameter("kind", "CombiningKind")], [], []))


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


def get_files_contents_as_json() -> str:
    all_files = []
    for file in get_tablegen_op_file_list():
        contents = get_file_contents(file)
        if contents is not None:
            all_files.append(contents)
    return json.dumps(all_files)


def get_stats_from_json(contents) -> Stats:
    contents = json.loads(contents)
    stats = Stats()
    for content in contents:
        stats.add_stats(Stats.from_json(content))
    add_cpp_types(stats)
    add_cpp_attributes(stats)
    remove_unnecessary_verifiers(stats)
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
        if isinstance(constraint, PredicateConstraint) and not constraint.is_declarative():
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
    type_res = get_global_type_distribution(stats, lambda type: int(all([p.is_declarative() for p in type.parameters])))[1]
    attr_res = get_global_attr_distribution(stats, lambda attr: int(all([p.is_declarative() for p in attr.parameters])))[1]
    # noinspection PyTypeChecker
    type_res = dict(filter(lambda x: sum(x[1]) != 0, type_res.items()))
    # noinspection PyTypeChecker
    attr_res = dict(filter(lambda x: sum(x[1]) != 0, attr_res.items()))
    print(f"type_params_decl = {type_res}")
    print(f"attr_params_decl = {attr_res}")


def create_type_verifiers_plot(stats: Stats):
    type_res = get_global_type_distribution(stats, lambda type: int(not type.hasVerifier))[1]
    attr_res = get_global_attr_distribution(stats, lambda attr: int(not attr.hasVerifier))[1]
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
    num_var_operands = get_global_op_distribution(stats, lambda x: sum([isinstance(operand.constraint, VariadicConstraint) for operand in x.operands]))
    print(f"num_var_operands_mean = {num_var_operands[0]}")
    print(f"num_var_operands_per_dialect = {num_var_operands[1]}")
    num_results = get_global_op_distribution(stats, lambda x: len(x.results))
    print(f"num_results_mean = {num_results[0]}")
    print(f"num_results_per_dialect = {num_results[1]}")
    num_var_results = get_global_op_distribution(stats, lambda x: sum([isinstance(result.constraint, VariadicConstraint) for result in x.results]))
    print(f"num_var_results_mean = {num_var_results[0]}")
    print(f"num_var_results_per_dialect = {num_var_results[1]}")
    num_attributes = get_global_op_distribution(stats, lambda x: len(x.attributes))
    print(f"num_attributes_mean = {num_attributes[0]}")
    print(f"num_attributes_per_dialect = {num_attributes[1]}")
    num_regions = get_global_op_distribution(stats, lambda x: len(x.regions))
    print(f"num_regions_mean = {num_regions[0]}")
    print(f"num_regions_per_dialect = {num_regions[1]}")
    has_verifiers = get_global_op_distribution(stats, lambda x: int(x.hasVerifier))
    print(f"has_verifier_mean = {has_verifiers[0]}")
    print(f"has_verifier_per_dialect = {has_verifiers[1]}")
    op_args_decl = get_global_op_distribution(stats, lambda x: int(x.is_operands_results_attrs_declarative()))
    print(f"op_args_decl_mean = {op_args_decl[0]}")
    print(f"op_args_decl_per_dialect = {op_args_decl[1]}")
    op_non_decl_constraints = get_constraints_culprits(stats)
    print(f"op_non_decl_constraints = {op_non_decl_constraints}")


def __main__():
    # stats = get_stat_from_files()

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
    print("total:", (len(stats.ops), sum([dialect.numOperations for dialect in stats.dialects.values()]),))

    print("Number of types defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.types), x.numTypes)))
    print("total:", (len(stats.types), sum([dialect.numTypes for dialect in stats.dialects.values()]),))

    print("Number of attributes defined in TableGen, and in total")
    print(get_dialect_values(stats, lambda x: (len(x.attrs), x.numAttributes)))
    print("total:", (len(stats.attrs), sum([dialect.numAttributes for dialect in stats.dialects.values()]),))

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

from analyze_tablegen.irdl import *
import os
import subprocess
import json

verifiers_allow_len_equality = False


# Some dialects are not included since they rely on generated tablegen files
def get_tablegen_op_file_list(llvm_root):
    res_files = []
    for root, dirs, files in os.walk(llvm_root + "/mlir/include"):
        for file in files:
            if file.endswith(".td"):
                res_files.append(os.path.join(root, file))
    return res_files


def get_file_contents(file, llvm_root, tblgen_extract_path) -> Optional[str]:
    root, file = os.path.split(file)
    res = subprocess.run([
        tblgen_extract_path,
        os.path.join(root, file), f"--I={llvm_root}/mlir/include",
        f"--I={root}"
    ],
                         capture_output=True)
    if res.returncode != 0:
        return None
    print("../build/bin/tblgen-extract", os.path.join(root, file),
          f"--I={llvm_root}/mlir/include", f"--I={root}")
    return json.loads(res.stdout)


def get_stat_from_file(file, llvm_root, tblgen_extract_path) -> Optional[Stats]:
    contents = get_file_contents(file, llvm_root, tblgen_extract_path)
    if contents is None:
        return None
    return Stats.from_json(contents)


def add_cpp_types(stats: Stats):
    # linalg
    stats.add_type(Type("range", "linalg", "UNKNOWN", False, [], [], []))

    # gpu
    stats.add_type(Type("async.token", "gpu", "UNKNOWN", False, [], [], []))
    # verifyCompatibleShape
    stats.add_type(
        Type("mma_matrix", "gpu", "UNKNOWN", True, [
            AttrOrTypeParameter("shape_x", "int64_t"),
            AttrOrTypeParameter("shape_y", "int64_t"),
            AttrOrTypeParameter("elementType", "Type"),
            AttrOrTypeParameter("operand", "StringAttr")
        ], [], []))
    # spv
    stats.add_type(
        Type("array", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("elementType", "Type"),
            AttrOrTypeParameter("elementCount", "unsigned"),
            AttrOrTypeParameter("stride", "unsigned")
        ], [], []))
    stats.add_type(
        Type("coopmatrix", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("elementType", "Type"),
            AttrOrTypeParameter("rows", "unsigned"),
            AttrOrTypeParameter("columns", "unsigned"),
            AttrOrTypeParameter("scope", "Scope")
        ], [], []))
    stats.add_type(
        Type("image", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("elementType", "Type"),
            AttrOrTypeParameter("dim", "Dim"),
            AttrOrTypeParameter("depthInfo", "ImageDepthInfo"),
            AttrOrTypeParameter("arrayedInfo", "ImageArrayedInfo"),
            AttrOrTypeParameter("samplingInfo", "ImageSamplingInfo"),
            AttrOrTypeParameter("samplerUseInfo", "ImageSamplerUseInfo"),
            AttrOrTypeParameter("format", "ImageFormat")
        ], [], []))
    stats.add_type(
        Type("ptr", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("pointeeType", "Type"),
            AttrOrTypeParameter("storageClass", "StorageClass")
        ], [], []))
    stats.add_type(
        Type("rtarray", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("elementType", "Type"),
            AttrOrTypeParameter("stride", "int")
        ], [], []))
    stats.add_type(
        Type("sampled_image", "spv", "UNKNOWN", False,
             [AttrOrTypeParameter("imageType", "Type")], [], []))
    stats.add_type(
        Type("struct", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("memberTypes", "ArrayRef<Type>"),
            AttrOrTypeParameter("offsetInfo", "ArrayRef<OffsetInfo>"),
            AttrOrTypeParameter("memberDecorations",
                                "ArrayRef<MemberDecorationInfo>")
        ], [], []))
    stats.add_type(
        Type("matrix", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("columnType", "Type"),
            AttrOrTypeParameter("columnCount", "uint32_t")
        ], [], []))

    # llvm
    stats.add_type(Type("void", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("ppc_fp128", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("x86mmx", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("token", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("label", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("metadata", "llvm", "UNKNOWN", False, [], [], []))
    stats.add_type(
        Type("func", "llvm", "UNKNOWN", False, [
            AttrOrTypeParameter("result", "Type"),
            AttrOrTypeParameter("arguments", "ArrayRef<Type>"),
            AttrOrTypeParameter("isVarArg", "bool")
        ], [], []))
    stats.add_type(
        Type("ptr", "llvm", "UNKNOWN", False, [
            AttrOrTypeParameter("pointee", "Type"),
            AttrOrTypeParameter("addressSpace", "unsigned")
        ], [], ["DataLayoutTypeInterface::Trait"]))
    # Check that a value is strictly positive
    stats.add_type(
        Type("fixed_vec", "llvm", "UNKNOWN", True, [
            AttrOrTypeParameter("elementType", "Type"),
            AttrOrTypeParameter("numElements", "unsigned")
        ], [], []))
    # Check that a value is strictly positive
    stats.add_type(
        Type("scalable_vec", "llvm", "UNKNOWN", True, [
            AttrOrTypeParameter("elementType", "Type"),
            AttrOrTypeParameter("numElements", "unsigned")
        ], [], []))
    stats.add_type(
        Type("array", "llvm", "UNKNOWN", False, [
            AttrOrTypeParameter("elementType", "Type"),
            AttrOrTypeParameter("numElements", "unsigned")
        ], [], []))
    # Complex underlying type that requires non-trivial verifier
    stats.add_type(
        Type("struct", "llvm", "UNKNOWN", True,
             [AttrOrTypeParameter("arg", "LLVMStruct")], [], []))

    # shape
    stats.add_type(Type("shape", "shape", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("size", "shape", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("value_shape", "shape", "UNKNOWN", False, [], [], []))
    stats.add_type(Type("witness", "shape", "UNKNOWN", False, [], [], []))

    # quant
    # Complex verifier
    stats.add_type(
        Type("any", "quant", "UNKNOWN", False, [
            AttrOrTypeParameter("flags", "unsigned"),
            AttrOrTypeParameter("storageType", "Type"),
            AttrOrTypeParameter("expressedType", "Type"),
            AttrOrTypeParameter("storageTypeMin", "int64_t"),
            AttrOrTypeParameter("storageTypeMax", "int64_t")
        ], [], []))
    # Complex verifier
    stats.add_type(
        Type("uniform", "quant", "UNKNOWN", False, [
            AttrOrTypeParameter("flags", "unsigned"),
            AttrOrTypeParameter("storageType", "Type"),
            AttrOrTypeParameter("expressedType", "Type"),
            AttrOrTypeParameter("scale", "double"),
            AttrOrTypeParameter("zeroPoint", "int64_t"),
            AttrOrTypeParameter("storageTypeMin", "int64_t"),
            AttrOrTypeParameter("storageTypeMax", "int64_t")
        ], [], []))
    # Complex verifier
    stats.add_type(
        Type("uniform_per_axis", "quant", "UNKNOWN", False, [
            AttrOrTypeParameter("flags", "unsigned"),
            AttrOrTypeParameter("storageType", "Type"),
            AttrOrTypeParameter("expressedType", "Type"),
            AttrOrTypeParameter("scales", "ArrayRef<double>"),
            AttrOrTypeParameter("zeroPoints", "ArrayRef<int64_t>"),
            AttrOrTypeParameter("quantizedDimension", "int64_t"),
            AttrOrTypeParameter("storageTypeMin", "int64_t"),
            AttrOrTypeParameter("storageTypeMax", "int64_t")
        ], [], []))
    # Less or equal comparison
    stats.add_type(
        Type("calibrated", "quant", "UNKNOWN", False, [
            AttrOrTypeParameter("expressedType", "Type"),
            AttrOrTypeParameter("min", "double"),
            AttrOrTypeParameter("max", "double")
        ], [], []))


def add_cpp_attributes(stats: Stats):
    # spv
    stats.add_attr(
        Attr("interface_var_abi", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("descriptorSet", "uint32_t"),
            AttrOrTypeParameter("binding", "uint32_t"),
            AttrOrTypeParameter("storageClass", "Optional<StorageClass>")
        ], [], []))
    stats.add_attr(
        Attr("ver_cap_ext", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("version", "Version"),
            AttrOrTypeParameter("capabilities", "ArrayRef<Capability>"),
            AttrOrTypeParameter("extensions", "ArrayRef<Extension>")
        ], [], []))
    stats.add_attr(
        Attr("target_env", "spv", "UNKNOWN", False, [
            AttrOrTypeParameter("triple", "VerCapExtAttr"),
            AttrOrTypeParameter("vendorID", "Vendor"),
            AttrOrTypeParameter("deviceType", "DeviceType"),
            AttrOrTypeParameter("deviceId", "uint32_t"),
            AttrOrTypeParameter("limits", "DictionaryAttr")
        ], [], []))

    # vector
    stats.add_attr(
        Attr("combining_kind", "vector", "UNKNOWN", False,
             [AttrOrTypeParameter("kind", "CombiningKind")], [], []))


def remove_unnecessary_verifiers(stats: Stats):
    # Types:
    stats.dialects["builtin"].types["Builtin_Complex"].hasVerifier = False
    stats.dialects["builtin"].types[
        "Builtin_UnrankedMemRef"].hasVerifier = False
    stats.dialects["builtin"].types[
        "Builtin_UnrankedTensor"].hasVerifier = False

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


def get_stat_from_files(llvm_root, tblgen_extract_path):
    stats = Stats()
    for file in get_tablegen_op_file_list(llvm_root):
        file_stats = get_stat_from_file(file, llvm_root, tblgen_extract_path)
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


def get_files_contents_as_json(llvm_root, tblgen_extract_path) -> str:
    all_files = []
    for file in get_tablegen_op_file_list(llvm_root):
        contents = get_file_contents(file, llvm_root, tblgen_extract_path)
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


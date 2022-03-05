from analyze_tablegen.irdl import *
import typing


class Simplifier(ABC):
    @staticmethod
    @abstractmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        ...

    @classmethod
    def simplify_or_copy(cls, constraint: Constraint) -> Constraint:
        res = cls.simplify(constraint)
        if res is None:
            return constraint
        return res


class CppClassToName(Simplifier):
    @staticmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        if not isinstance(constraint, CppBaseConstraint):
            return None
        if constraint.name == "::mlir::VectorType":
            return BaseConstraint("builtin", "vector")
        if constraint.name == "::mlir::TensorType":
            return BaseConstraint("builtin", "tensor")
        if constraint.name == "::mlir::IndexType":
            return BaseConstraint("builtin", "index")
        if constraint.name == "::mlir::IntegerType":
            return BaseConstraint("builtin", "integer")
        if constraint.name == "::mlir::MemRefType":
            return BaseConstraint("builtin", "memref")
        if constraint.name == "::mlir::UnitAttr":
            return BaseConstraint("builtin", "unit")
        if constraint.name == "::mlir::LLVM::LLVMPointerType":
            return BaseConstraint("llvm", "ptr")
        if constraint.name == "::mlir::LLVM::LLVMVoidType":
            return BaseConstraint("llvm", "void")
        if constraint.name == "::mlir::LLVM::LLVMPPCFP128Type":
            return BaseConstraint("llvm", "ppc_fp128")
        if constraint.name == "::mlir::LLVM::LLVMX86MMXType":
            return BaseConstraint("llvm", "x86_mmx")
        if constraint.name == "::mlir::LLVM::LLVMTokenType":
            return BaseConstraint("llvm", "token")
        if constraint.name == "::mlir::LLVM::LLVMLabelType":
            return BaseConstraint("llvm", "label")
        if constraint.name == "::mlir::LLVM::LLVMMetadataType":
            return BaseConstraint("llvm", "metadata")
        if constraint.name == "::mlir::LLVM::LLVMFunctionType":
            return BaseConstraint("llvm", "func")
        if constraint.name == "::mlir::LLVM::LLVMArrayType":
            return BaseConstraint("llvm", "array")
        if constraint.name == "::mlir::LLVM::LLVMStructType":
            return BaseConstraint("llvm", "struct")
        if constraint.name == "::mlir::LLVM::LLVMScalableVectorType":
            return BaseConstraint("llvm", "vec")
        if constraint.name == "::mlir::arm_sve::ScalableVectorType":
            return BaseConstraint("arm_sve", "vector")
        if constraint.name == "::mlir::FloatType":
            return OrConstraint([BaseConstraint("builtin", "bf16"),
                                 BaseConstraint("builtin", "f16"),
                                 BaseConstraint("builtin", "f32"),
                                 BaseConstraint("builtin", "f64"),
                                 BaseConstraint("builtin", "f80"),
                                 BaseConstraint("builtin", "f128")])
        if constraint.name == "::mlir::ComplexType":
            return BaseConstraint("builtin", "complex")
        if constraint.name == "::mlir::ArrayAttr":
            return BaseConstraint("builtin", "array")
        if constraint.name == "::mlir::AffineMapAttr":
            return BaseConstraint("builtin", "affine_map")
        if constraint.name == "::mlir::async::TokenType":
            return BaseConstraint("async", "token")
        if constraint.name == "::mlir::async::ValueType":
            return BaseConstraint("async", "value")
        if constraint.name == "::mlir::async::GroupType":
            return BaseConstraint("async", "group")
        if constraint.name == "::mlir::IntegerAttr":
            return BaseConstraint("builtin", "integer")
        if constraint.name == "::mlir::gpu::AsyncTokenType":
            return BaseConstraint("gpu", "async.token")
        if constraint.name == "::mlir::StringAttr":
            return BaseConstraint("builtin", "string")
        if constraint.name == "::mlir::BoolAttr":
            return BaseConstraint("builtin", "bool")
        if constraint.name == "::mlir::TypeAttr":
            return BaseConstraint("builtin", "type")
        if constraint.name == "::mlir::UnrankedTensorType":
            return BaseConstraint("builtin", "tensor")
        if constraint.name == "::mlir::FloatAttr":
            return BaseConstraint("builtin", "float")
        return None


class MergeOrAnd(Simplifier):
    @staticmethod
    def should_merge(typ: Union[typing.Type[AndConstraint],
                                typing.Type[OrConstraint]],
                     constraint: Union[AndConstraint, OrConstraint]) -> bool:
        for operand in constraint.operands:
            if isinstance(operand, typ):
                return True
        return False

    @staticmethod
    def merge(typ: Union[typing.Type[AndConstraint],
                         typing.Type[OrConstraint]],
              constraints: List[Constraint]) -> Constraint:
        new_operands = []
        for constraint in constraints:
            if isinstance(constraint, typ):
                for sub_operand in constraint.operands:
                    new_operands.append(sub_operand)
            else:
                new_operands.append(constraint)
        # noinspection PyArgumentList
        return typ(new_operands)

    @staticmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        def simplify_(
            typ: Union[typing.Type[AndConstraint], typing.Type[OrConstraint]]
        ) -> Optional[Constraint]:
            if isinstance(constraint, typ):
                if not MergeOrAnd.should_merge(typ, constraint):
                    return None
                return MergeOrAnd.merge(typ, constraint.operands)
            return None

        simplify_and = simplify_(AndConstraint)
        if simplify_and is not None:
            return simplify_and
        return simplify_(OrConstraint)


class AndRedundantSimplifier(Simplifier):
    @staticmethod
    def is_redundant(constraint: Constraint, other: Constraint) -> bool:
        if constraint == other:
            return True
        if isinstance(constraint, BaseConstraint):
            if isinstance(
                    other, ParametricTypeConstraint
            ) and other.type == constraint.name and other.dialect == constraint.dialect:
                return True
            return False

    @staticmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        if not isinstance(constraint, AndConstraint):
            return None
        to_remove = set()
        for idx, sub_constraint in enumerate(constraint.operands):
            if isinstance(sub_constraint, AnyConstraint):
                to_remove.add(idx)
                continue
            for idx2, sub_constraint2 in enumerate(constraint.operands):
                if idx == idx2 or idx2 in to_remove:
                    continue
                if AndRedundantSimplifier.is_redundant(sub_constraint,
                                                       sub_constraint2):
                    to_remove.add(idx)
                    continue

        if len(to_remove) == 0:
            return None

        new_operands = []
        for idx, operand in enumerate(constraint.operands):
            if idx not in to_remove:
                new_operands.append(operand)
        return AndConstraint(new_operands)


class AndFusionSimplifier(Simplifier):
    @staticmethod
    def try_simplify_one(constraint: Constraint,
                         given: Constraint) -> Optional[Constraint]:
        if isinstance(given, BaseConstraint) and isinstance(
                constraint, ShapedTypeConstraint):
            if given.name == "vector" and given.dialect == "builtin":
                res = ParametricTypeConstraint(
                    "builtin", "vector",
                    [AnyConstraint(), constraint.elemTypeConstraint])
                return simplify_constraints_until_convergence(res)
            if given.name == "tensor" and given.dialect == "builtin":
                res = ParametricTypeConstraint("builtin", "tensor", [
                    AnyConstraint(), constraint.elemTypeConstraint,
                    AnyConstraint()
                ])
                return simplify_constraints_until_convergence(res)
        if isinstance(given, ShapedTypeConstraint) and isinstance(
                constraint, ShapedTypeConstraint):
            sub_constraint = AndConstraint(
                [given.elemTypeConstraint, constraint.elemTypeConstraint])
            sub_constraint = simplify_constraints_until_convergence(
                sub_constraint)
            return ShapedTypeConstraint(sub_constraint)

    @staticmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        if not isinstance(constraint, AndConstraint):
            return None
        for i in range(len(constraint.operands)):
            for j in range(len(constraint.operands)):
                if i == j:
                    continue
                res = AndFusionSimplifier.try_simplify_one(
                    constraint.operands[i], constraint.operands[j])
                if res is not None:
                    if i < j:
                        return AndConstraint(constraint.operands[:i] + [res] +
                                             constraint.operands[i + 1:j] +
                                             constraint.operands[j + 1:])
                    if i > j:
                        return AndConstraint(constraint.operands[:j] +
                                             constraint.operands[j + 1:i] +
                                             [res] +
                                             constraint.operands[i + 1:])

        return None


class TrivialAndOrSimplifier(Simplifier):
    @staticmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        if isinstance(constraint, AndConstraint) or isinstance(
                constraint, OrConstraint):
            if len(constraint.operands) == 1:
                return constraint.operands[0]
        return None


class FactorizeOrSimplifier(Simplifier):
    @staticmethod
    def can_merge_parametric(
            constraint1: ParametricTypeConstraint,
            constraint2: ParametricTypeConstraint) -> Optional[Constraint]:
        if len(constraint1.params) != len(constraint2.params):
            return None

        def can_merge_parametric_(index: int) -> bool:
            for k in range(len(constraint1.params)):
                if k == index:
                    continue
                if constraint1.params[k] != constraint2.params[k]:
                    return False
            return True

        for k in range(len(constraint1.params)):
            if can_merge_parametric_(k):
                operand_index = k
                break
        else:
            return None

        or_constraint = OrConstraint([
            constraint1.params[operand_index],
            constraint2.params[operand_index]
        ])
        or_constraint = simplify_constraints_until_convergence(or_constraint)
        return ParametricTypeConstraint(
            constraint1.dialect, constraint1.type,
            constraint1.params[0:operand_index] + [or_constraint] +
            constraint1.params[operand_index + 1:])

    @staticmethod
    def can_merge(constraint1: Constraint,
                  constraint2: Constraint) -> Optional[Constraint]:
        if isinstance(constraint1, ParametricTypeConstraint) and isinstance(
                constraint2, ParametricTypeConstraint):
            res = FactorizeOrSimplifier.can_merge_parametric(
                constraint1, constraint2)
            if res is None:
                return None
            return res
        if isinstance(constraint1, ShapedTypeConstraint) and isinstance(
                constraint2, ShapedTypeConstraint):
            sub_constr = OrConstraint([
                constraint1.elemTypeConstraint, constraint2.elemTypeConstraint
            ])
            sub_constr = simplify_constraints_until_convergence(sub_constr)
            return ShapedTypeConstraint(sub_constr)
        return None

    @staticmethod
    def simplify_once(constraints: List[Constraint]) -> Optional[Constraint]:
        for i in range(len(constraints)):
            constraint_i = constraints[i]
            for j in range(i + 1, len(constraints)):
                constraint_j = constraints[j]
                merge_res = FactorizeOrSimplifier.can_merge(
                    constraint_i, constraint_j)
                if merge_res is None:
                    continue
                return OrConstraint(constraints[:i] + [merge_res] +
                                    constraints[i + 1:j] + constraints[j + 1:])
        return None

    @staticmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        if not isinstance(constraint, OrConstraint):
            return None
        new_constraint = FactorizeOrSimplifier.simplify_once(
            constraint.operands)
        if new_constraint is None:
            return None
        return new_constraint


all_simplifiers = [
    CppClassToName, MergeOrAnd, AndRedundantSimplifier, TrivialAndOrSimplifier,
    FactorizeOrSimplifier, AndFusionSimplifier
]


def simplify_constraints_until_convergence(
        constraint: Constraint) -> Constraint:
    for simplifier in all_simplifiers:
        new_constraint = simplifier.simplify(constraint)
        if new_constraint is not None:
            return simplify_constraints_until_convergence(new_constraint)
    return constraint


def simplify(stats: Stats) -> Stats:
    return stats.map_constraints(simplify_constraints_until_convergence)

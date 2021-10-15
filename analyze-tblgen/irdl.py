from __future__ import annotations
from utils import *
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import *

indent_size = 2


@dataclass
class Trait:
    @staticmethod
    def from_json(json):
        if json["kind"] == "native":
            return NativeTrait.from_json(json)
        if json["kind"] == "pred":
            return PredTrait.from_json(json)
        if json["kind"] == "internal":
            return InternalTrait.from_json(json)
        assert False

    def is_declarative(self) -> bool:
        raise NotImplemented


@from_json
class NativeTrait(Trait):
    name: str

    def is_declarative(self) -> bool:
        # Do not have verifiers. Note, we may need to add dependencies between traits
        if self.name == "::mlir::OpTrait::IsCommutative":
            return True
        if self.name == "::mlir::OpTrait::Scalarizable":
            return True
        if self.name == "::mlir::OpTrait::Vectorizable":
            return True
        if self.name == "::mlir::OpTrait::Tensorizable":
            return True
        if self.name == "::mlir::OpTrait::spirv::UsableInSpecConstantOp":
            return True
        if self.name == "::mlir::OpTrait::spirv::UnsignedOp":
            return True
        if self.name == "::mlir::OpTrait::spirv::SignedOp":
            return True
        if self.name == "::mlir::OpTrait::HasRecursiveSideEffects":
            return True
        if self.name == "::mlir::OpTrait::MemRefsNormalizable":
            return True

        # Have verifiers, but should be builtins
        if self.name == "::mlir::OpTrait::IsTerminator":
            return True
        m = re.compile(r"::mlir::OpTrait::HasParent<(.*)>::Impl").match(self.name)
        if m is not None:
            return True
        if self.name == "::mlir::OpTrait::IsIsolatedFromAbove":
            return True

        # Are replaced by IRDL way of doing things
        if self.name == "::mlir::OpTrait::SameOperandsAndResultType":
            return True
        if self.name == "::mlir::OpTrait::SameTypeOperands":
            return True
        if self.name == "::mlir::OpTrait::Elementwise":
            return True
        m = re.compile(r"::mlir::OpTrait::SingleBlockImplicitTerminator<(.*)>::Impl").match(self.name)
        if m is not None:
            return True
        if self.name == "::mlir::OpTrait::AttrSizedOperandSegments":
            return True

        # Cannot be replaced by IRDL for now
        if self.name == "::mlir::OpTrait::SameOperandsAndResultShape":
            return False

        return False


@from_json
class PredTrait(Trait):
    pred: str

    def is_declarative(self) -> bool:
        if re.compile(
                r"\(getElementTypeOrSelf\(\$_op.getResult\((.*)\)\) == getElementTypeOrSelf\(\$_op.getOperand\((.*)\)\)\)").match(
            self.pred) is not None:
            return True
        if self.pred == "(std::equal_to<>()($tensor.getType().cast<ShapedType>().getElementType(), $result.getType()))":
            return True
        return False


@from_json
class InternalTrait(Trait):
    name: str

    def is_declarative(self) -> bool:
        return False


@dataclass(eq=True, unsafe_hash=True)
class Constraint(ABC):
    @staticmethod
    def from_predicate(predicate: str) -> Constraint:
        predicate = simplify_expression(predicate)

        m = re.compile(r"\$_self.isa<(.*)>\(\)").match(predicate)
        if m is not None:
            return BaseConstraint(predicate[11:-3])

        m = re.compile(r"!\((.*)\)").match(predicate)
        if m is not None:
            constraint = Constraint.from_predicate(m.group(0)[2:-1])
            return NotConstraint(constraint)

        and_operands = separate_on_operator(predicate, "&&")
        if and_operands is not None:
            operand1 = Constraint.from_predicate(and_operands[0])
            operand2 = Constraint.from_predicate(and_operands[1])
            return AndConstraint(operand1, operand2)

        or_operands = separate_on_operator(predicate, "||")
        if or_operands is not None:
            operand1 = Constraint.from_predicate(or_operands[0])
            operand2 = Constraint.from_predicate(or_operands[1])
            return OrConstraint([operand1, operand2])

        m = re.compile(r"!(.*)").match(predicate)
        if m is not None:
            operand = Constraint.from_predicate(m.group(1))
            return NotConstraint(operand)

        if predicate == "true":
            return AnyConstraint()

        m = re.match(r"\$_self.isInteger\((.*)\)", predicate)
        if m is not None:
            return ParametricTypeConstraint("Builtin_Integer", [IntEqConstraint(int(m.group(1))),
                                                                AnyConstraint()])

        m = re.match(r"\$_self.isSignlessInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraint("Builtin_Integer", [AnyConstraint(),
                                                                    EnumValueEqConstraint("Signless")])
            return ParametricTypeConstraint("Builtin_Integer", [IntEqConstraint(int(val)),
                                                                EnumValueEqConstraint("Signless")])

        m = re.match(r"\$_self.isUnsignedInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraint("Builtin_Integer", [AnyConstraint(),
                                                                    EnumValueEqConstraint("Unsigned")])
            return ParametricTypeConstraint("Builtin_Integer", [IntEqConstraint(int(val)),
                                                                EnumValueEqConstraint("Unsigned")])

        m = re.match(r"\$_self.isSignedInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraint("Builtin_Integer", [AnyConstraint(),
                                                                    EnumValueEqConstraint("Signed")])
            return ParametricTypeConstraint("Builtin_Integer", [IntEqConstraint(int(val)),
                                                                EnumValueEqConstraint("Signed")])

        if predicate == "$_self.isBF16()":
            return ParametricTypeConstraint("Builtin_BFloat16", [])

        m = re.match(r"\$_self.isF(.*)\(\)", predicate)
        if m is not None:
            return ParametricTypeConstraint("Builtin_Float" + m.group(1), [])

        if predicate == "$_self.isIndex()":
            return ParametricTypeConstraint("Builtin_Index", [])

        m = re.match(r"\$_self.cast<::mlir::FloatAttr>\(\).getType\(\)(.*)", predicate)
        if m is not None:
            type_predicate = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("Builtin_FloatAttr", [type_predicate, AnyConstraint()])

        m = re.match(r"\$_self.cast<::mlir::ShapedType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            element_type_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ShapedTypeConstraint(element_type_constraint)

        m = re.match(r"::mlir::LLVM::getVectorElementType\(\$_self\)(.*)", predicate)
        if m is not None:
            element_type_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return LLVMVectorOfConstraint(element_type_constraint)

        m = re.match(r"\$_self.cast<::mlir::DenseIntElementsAttr>\(\)( ?).getType\(\)( ?).getElementType\(\)( ?)(.*)",
                     predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(4))
            return ParametricTypeConstraint("DenseIntElementsAttr", [sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::arm_sve::ScalableVectorType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("ScalableVectorType", [AnyConstraint(), sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::LLVM::LLVMPointerType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("ptr", [sub_constraint, AnyConstraint()])

        m = re.match(r"\$_self.cast<::mlir::spirv::CooperativeMatrixNVType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("coopmatrix",
                                            [sub_constraint, AnyConstraint(), AnyConstraint(),
                                             AnyConstraint()])

        m = re.match(r"\$_self.cast<::mlir::pdl::RangeType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("PDL_Range", [sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::gpu::MMAMatrixType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("PDL_Range",
                                            [AnyConstraint(), AnyConstraint(), sub_constraint,
                                             AnyConstraint()])

        m = re.match(r"\$_self.cast<::mlir::ComplexType>\(\).getElementType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("Complex", [sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::IntegerAttr>\(\).getType\(\)(.*)", predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("Builtin_IntegerAttr", [sub_constraint, AnyConstraint()])

        m = re.match(
            r"::llvm::all_of\(\$_self.cast<::mlir::ArrayAttr>\(\), \[&]\(::mlir::Attribute attr\) { return (.*); }\)",
            predicate)
        if m is not None:
            group = re.sub(r"attr\.", "$_self.", m.group(1))
            sub_constraint = Constraint.from_predicate(group)
            return AttrArrayOf(sub_constraint)

        m = re.match(
            r"::llvm::all_of\(\$_self.cast<::mlir::TupleType>\(\).getTypes\(\), \[]\(Type t\) { return (.*); }\)",
            predicate)
        if m is not None:
            group = re.sub(r"t\.", "$_self.", m.group(1))
            sub_constraint = Constraint.from_predicate(group)
            return TupleOf(sub_constraint)

        m = re.match(r"\$_self.cast<(::mlir::)?StringAttr>\(\).getValue\(\) == \"(.*)\"", predicate)
        if m is not None:
            str_val = m.group(1)
            return ParametricTypeConstraint("Builtin_StringAttr", [StringEqConstraint(str_val)])

        llvm_float_types = ["Builtin_BFloat16", "Builtin_Float16", "Builtin_Float32", "Builtin_Float64",
                            "Builtin_Float80", "Builtin_Float128", "ppc_fp128"]

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType($_self)":
            return OrConstraint([BaseConstraint(typ) for typ in llvm_float_types])

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))":
            element_type_constraint = OrConstraint([BaseConstraint(typ) for typ in llvm_float_types])
            return LLVMVectorOfConstraint(element_type_constraint)

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType($_self.cast<::mlir::LLVM::LLVMPointerType>().getElementType())":
            sub_constraint = OrConstraint([BaseConstraint(typ) for typ in llvm_float_types])
            return ParametricTypeConstraint("ptr", [sub_constraint, AnyConstraint()])

        if predicate == "::mlir::LLVM::isCompatibleType($_self)":
            return LLVMCompatibleType()

        if predicate == "::mlir::LLVM::isCompatibleType($_self.cast<::mlir::LLVM::LLVMPointerType>().getElementType())":
            return ParametricTypeConstraint("ptr", [LLVMCompatibleType(), AnyConstraint()])

        if predicate == "::mlir::LLVM::isCompatibleVectorType($_self)":
            llvm_vector_types: List[Constraint] = [BaseConstraint("fixed_vec"),
                                                   BaseConstraint("scalable_vec")]
            vector_elem_float_types = ["Builtin_BFloat16", "Builtin_Float16", "Builtin_Float32", "Builtin_Float64",
                                       "Builtin_Float80", "Builtin_Float128", "ppc_fp128"]
            signless_integer = ParametricTypeConstraint("Builtin_Integer",
                                                        [AnyConstraint(),
                                                         EnumValueEqConstraint("Signless")])
            vector_elem_types = OrConstraint(
                [BaseConstraint(typ) for typ in vector_elem_float_types] + [signless_integer])
            vector_type = ParametricTypeConstraint("vector", [ArrayRefConstraint([AnyConstraint()]),
                                                              vector_elem_types])
            return OrConstraint(llvm_vector_types + [vector_type])

        if predicate == "$_self.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()":
            return BaseConstraint("::mlir::TypeAttr")

        return PredicateConstraint.from_predicate(predicate)

    @staticmethod
    def from_json(json):
        if json["kind"] == "variadic":
            return VariadicConstraint.from_json(json)
        if json["kind"] == "optional":
            return OptionalConstraint.from_json(json)
        if json["kind"] == "typeDef":
            return TypeDefConstraint.from_json(json)
        if json["kind"] == "integer":
            return IntegerConstraint.from_json(json)
        assert json["kind"] == "predicate"
        return Constraint.from_predicate(json["predicate"])

    @abstractmethod
    def is_declarative(self) -> bool:
        ...

    @abstractmethod
    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        ...

    def walk_constraints(self, func: Callable[[Constraint]]):
        func(self)
        self.walk_sub_constraints(func)


@from_json
class VariadicConstraint(Constraint):
    baseType: Constraint

    def is_declarative(self) -> bool:
        return self.baseType.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.baseType.walk_constraints(func)

    def __str__(self):
        return f"Variadic<{self.baseType}>"


@from_json
class OptionalConstraint(Constraint):
    baseType: Constraint

    def is_declarative(self) -> bool:
        return self.baseType.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.baseType.walk_sub_constraints(func)

    def __str__(self):
        return f"Optional<{self.baseType}>"


@from_json
class TypeDefConstraint(Constraint):
    dialect: str
    name: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f"{self.dialect}.{self.name}"


@from_json
class IntegerConstraint(Constraint):
    bitwidth: int

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f"IntegerOfSize<{self.bitwidth}>"


@dataclass(eq=False)
class AnyConstraint(Constraint):
    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f"Any"


@dataclass(eq=False)
class BaseConstraint(Constraint):
    name: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f"{self.name}"


@dataclass(eq=False)
class ParametricTypeConstraint(Constraint):
    base: str
    params: List[Constraint]

    def is_declarative(self) -> bool:
        return all(param.is_declarative() for param in self.params)

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        for param in self.params:
            param.walk_constraints(func)

    def __str__(self):
        return f"{self.base}<{', '.join([str(param) for param in self.params])}>"


@dataclass(eq=False)
class NotConstraint(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def __str__(self):
        return f"Not<{self.constraint}>"


@dataclass(eq=False)
class AndConstraint(Constraint):
    operand1: Constraint
    operand2: Constraint

    def is_declarative(self) -> bool:
        return self.operand1.is_declarative() and self.operand2.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.operand1.walk_constraints(func)
        self.operand2.walk_constraints(func)

    def __str__(self):
        return f"And<{self.operand1}, {self.operand2}>"


@dataclass(eq=False)
class OrConstraint(Constraint):
    operands: List[Constraint]

    def __init__(self, operands: List[Constraint]):
        self.operands = []
        for operand in operands:
            if isinstance(operand, OrConstraint):
                for sub_operand in operand.operands:
                    self.operands.append(sub_operand)
            else:
                self.operands.append(operand)

    def is_declarative(self) -> bool:
        return all([c.is_declarative for c in self.operands])

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        for operand in self.operands:
            operand.walk_constraints(func)

    def __str__(self):
        return f"AnyOf<{', '.join([str(operand) for operand in self.operands])}>"


@dataclass(eq=False)
class NotConstraint(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def __str__(self):
        return f"Not<{self.constraint}>"


@dataclass(eq=False)
class ShapedTypeConstraint(Constraint):
    elemTypeConstraint: Constraint

    def is_declarative(self) -> bool:
        return self.elemTypeConstraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.elemTypeConstraint.walk_constraints(func)

    def __str__(self):
        return f"ShapedTypeOf<{self.elemTypeConstraint}>"


@dataclass(eq=False)
class LLVMVectorOfConstraint(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def __str__(self):
        return f"LLVMVectorOf<{self.constraint}>"


@dataclass(eq=False)
class AttrArrayOf(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def __str__(self):
        return f"AttrArrayOf<{self.constraint}>"


@dataclass(eq=False)
class TupleOf(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def __str__(self):
        return f"TupleOf<{self.constraint}>"


@dataclass(eq=False)
class LLVMCompatibleType(Constraint):
    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f"LLVMCompatibleType"


@dataclass(eq=False)
class EnumValueEqConstraint(Constraint):
    value: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f'"{self.value}"'


@dataclass(eq=False)
class IntEqConstraint(Constraint):
    value: int

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f"{self.value}"


@dataclass(eq=False)
class StringEqConstraint(Constraint):
    value: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f"{self.value}"


@dataclass(eq=False)
class ArrayRefConstraint(Constraint):
    constraints: List[Constraint]

    def is_declarative(self) -> bool:
        return all([c.is_declarative() for c in self.constraints])

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        for constraint in self.constraints:
            constraint.walk_constraints(func)

    def __str__(self):
        return f"[{', '.join([str(c) for c in self.constraints])}]"


@from_json
class PredicateConstraint(Constraint):
    predicate: str

    @staticmethod
    def from_predicate(predicate: str) -> PredicateConstraint:
        return PredicateConstraint(predicate)

    def is_declarative(self) -> bool:
        self.predicate = simplify_expression(self.predicate)
        m = re.compile(r"\$_self.cast<(.*)>\(\).getRank\(\) == (.*)").match(self.predicate)
        if m is not None:
            return True

        if self.predicate == "$_self.cast<::mlir::ShapedType>().hasRank()":
            return True

        if re.compile(r"\$_self.cast<::mlir::ArrayAttr>\(\).size\(\) == (.*)").match(self.predicate) is not None:
            return True

        if re.compile(r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\)\[0].isa<(.*)>\(\)").match(
                self.predicate) is not None:
            return True

        if re.compile(
                r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\)\[1].isSignlessInteger\((.*)\)").match(
            self.predicate) is not None:
            return True

        if re.compile("::mlir::spirv::symbolize(.*)").match(self.predicate):
            return True

        if re.compile(r"\$_self.cast<mlir::quant::QuantizedType>\(\).getStorageTypeIntegralWidth\(\) == (.*)").match(
                self.predicate) is not None:
            return True

        if self.predicate == "$_self.cast<::mlir::TypeAttr>().getValue().isa<::mlir::MemRefType>()":
            return True

        # Harder cases:
        if self.predicate == "$_self.cast<::mlir::ArrayAttr>().size() <= 4":
            return False

        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getInt() >= 0":
            return False

        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getInt() <= 3":
            return False

        if self.predicate == "$_self.cast<IntegerAttr>().getValue().isStrictlyPositive()":
            return False

        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getValue().isNegative()":
            return False

        m = re.compile(r"\$_self.cast<(.*)>\(\).getNumElements\(\) == (.*)").match(self.predicate)
        if m is not None:
            return False

        m = re.compile(r"\$_self.cast<(.*)>\(\).getNumElements\(\) == (.*)").match(self.predicate)
        if m is not None:
            return False

        m = re.compile(r"\$_self.cast<(.*)>\(\).hasStaticShape\(\)").match(self.predicate)
        if m is not None:
            return False

        if self.predicate == "isStrided($_self.cast<::mlir::MemRefType>())":
            return False

        m = re.compile(r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\).size\(\) == (.*)").match(
            self.predicate)
        if m is not None:
            return False

        m = re.compile(r"(.*).cast<::mlir::LLVM::LLVMStructType>\(\).isOpaque\(\)").match(self.predicate)
        if m is not None:
            return False

        print(self.predicate)
        assert False

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def __str__(self):
        return f"Predicate<\"{self.predicate}\">"


@from_json
class NamedConstraint:
    name: str
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def __str__(self):
        return f"{self.name}: {self.constraint}"


@from_json
class Op:
    name: str
    dialect: str
    hasVerifier: bool
    numOperands: int
    numVariableLengthOperands: int
    numResults: int
    numRegions: int
    hasNoVariadicRegions: bool
    numSuccessors: int
    hasAssemblyFormat: bool
    operands: List[NamedConstraint]
    results: List[NamedConstraint]
    attributes: Dict[str, Constraint]
    traits: List[Trait]
    interfaces: List[str]

    def is_operands_results_attrs_declarative(self) -> bool:
        for operand in self.operands:
            if not operand.is_declarative():
                return False
        for result in self.results:
            if not result.is_declarative():
                return False
        for name, attr in self.attributes.items():
            if not attr.is_declarative():
                return False
        return True

    def is_traits_declarative(self) -> bool:
        for trait in self.traits:
            if not trait.is_declarative():
                return False
        return True

    def is_interface_declarative(self) -> bool:
        return not len(self.interfaces) > 0

    def is_declarative(self, check_traits: bool = True, check_interfaces: bool = True) -> bool:
        if self.hasVerifier:
            return False
        if not self.is_operands_results_attrs_declarative():
            return False
        if check_traits and not self.is_traits_declarative():
            return False
        if check_interfaces and not self.is_interface_declarative():
            return False
        return True

    def walk_constraints(self, func: Callable[[Constraint]]):
        for operand in self.operands:
            operand.constraint.walk_constraints(func)
        for result in self.results:
            result.constraint.walk_constraints(func)
        for attribute in self.attributes.values():
            attribute.walk_constraints(func)

    def print(self, indent_level=0):
        print(f"{' ' * indent_level}Operation {self.name} {{")

        # Operands
        if len(self.operands) != 0:
            print(f"{' ' * (indent_level + indent_size)}Operands (", end='')
            print(f",\n{' ' * (indent_level + indent_size + len('Operands ('))}".join(
                [str(operand) for operand in self.operands]), end='')
            print(")")

        # Results
        if len(self.results) != 0:
            print(f"{' ' * (indent_level + indent_size)}Results (", end='')
            print(f",\n{' ' * (indent_level + indent_size + len('Results ('))}".join(
                [str(result) for result in self.results]), end='')
            print(")")

        # Attributes
        if len(self.attributes) != 0:
            print(f"{' ' * (indent_level + indent_size)}Attributes (", end='')
            print(f",\n{' ' * (indent_level + indent_size + len('Attributes ('))}".join(
                [f'{name}: {attr}' for name, attr in self.attributes.items()]), end='')
            print(")")

        print(f"{' ' * indent_level}}}")


@from_json
class AttrOrTypeParameter:
    name: str
    cppType: str

    def get_group(self):
        base_names = ["Type", "::mlir::Type", "Attribute", "ShapedType", "DenseElementsAttr",
                      "DenseIntElementsAttr", "StringAttr", "VerCapExtAttr", "DictionaryAttr"]
        if self.cppType in base_names:
            return "type/attr"

        base_array_names = ["::llvm::ArrayRef<Attribute>", "::llvm::ArrayRef<NamedAttribute>", "ArrayRef<Type>",
                            "::llvm::ArrayRef<FlatSymbolRefAttr>"]
        if self.cppType in base_array_names:
            return "attr/type array"

        integer_names = ["unsigned", "uintptr_t", "int64_t", "uint32_t", "int", "APInt", "bool"]
        if self.cppType in integer_names:
            return "integer"

        int_array_names = ["::llvm::ArrayRef<int64_t>", "ArrayRef<int64_t>"]
        if self.cppType in int_array_names:
            return "integer array"

        float_names = ["double", "::llvm::APFloat"]
        if self.cppType in float_names:
            return "float"

        float_array_names = ["ArrayRef<double>"]
        if self.cppType in float_array_names:
            return "float array"

        string_names = ["Identifier", "::llvm::StringRef"]
        if self.cppType in string_names:
            return "string"

        string_array_names = ["ArrayRef<char>", "ArrayRef<StringRef>"]
        if self.cppType in string_array_names:
            return "string array"

        enum_names = ["Scope", "Dim", "ImageDepthInfo", "ImageArrayedInfo", "ImageSamplingInfo", "ImageSamplerUseInfo",
                      "ImageFormat", "StorageClass", "Optional<StorageClass>",
                      "Version", "Capability", "Extension", "Vendor",
                      "DeviceType", "CombiningKind", "SignednessSemantics", "FastmathFlags"]
        if self.cppType in enum_names:
            return "enum"

        enum_array_names = ["ArrayRef<OffsetInfo>", "::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>>",
                            "ArrayRef<MemberDecorationInfo>",
                            "::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType>",
                            "ArrayRef<Capability>", "ArrayRef<Extension>"]
        if self.cppType in enum_array_names:
            return "enum array"

        other_names = ["TypeID", "Location", "::llvm::ArrayRef<Location>", "AffineMap", "::llvm::ArrayRef<AffineMap>",
                       "IntegerSet", "TODO"]
        if self.cppType in other_names:
            return "other"

        assert False

    def is_declarative(self, builtins=True, enums=True):
        assert (not enums or builtins)
        base_names = ["Type", "::mlir::Type", "Attribute", "ShapedType", "DenseElementsAttr",
                      "DenseIntElementsAttr", "StringAttr", "VerCapExtAttr", "DictionaryAttr"]
        if self.cppType in base_names:
            return True

        # integers and floats
        builtin_names = ["unsigned", "uintptr_t", "int64_t", "uint32_t", "int", "APInt", "::llvm::APFloat",
                         "bool", "double", "Identifier", "::llvm::StringRef"]
        if self.cppType in builtin_names:
            return builtins

        # arrays
        arrays = ["::llvm::ArrayRef<int64_t>", "::llvm::ArrayRef<Attribute>", "::llvm::ArrayRef<NamedAttribute>",
                  "ArrayRef<Type>", "ArrayRef<char>",
                  "ArrayRef<StringRef>", "::llvm::ArrayRef<FlatSymbolRefAttr>", "ArrayRef<double>", "ArrayRef<int64_t>"]
        if self.cppType in arrays:
            return builtins

        # Enums
        enum_names = ["Scope", "Dim", "ImageDepthInfo", "ImageArrayedInfo", "ImageSamplingInfo", "ImageSamplerUseInfo",
                      "ImageFormat", "StorageClass", "Optional<StorageClass>", "ArrayRef<OffsetInfo>",
                      "::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>>", "ArrayRef<MemberDecorationInfo>",
                      "::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType>",
                      "Version", "Capability", "ArrayRef<Capability>", "Extension", "ArrayRef<Extension>", "Vendor",
                      "DeviceType", "CombiningKind", "SignednessSemantics", "FastmathFlags", "linkage::Linkage"]
        if self.cppType in enum_names:
            return enums

        if self.cppType == "TypeID":
            return False
        if self.cppType == "Location":
            return False
        if self.cppType == "::llvm::ArrayRef<Location>":
            return False
        if self.cppType == "AffineMap":
            return False
        if self.cppType == "::llvm::ArrayRef<AffineMap>":
            return False
        if self.cppType == "IntegerSet":
            return False
        if self.cppType == "LLVMStruct":
            return False

        print(self.cppType)
        assert False


@from_json
class Type:
    name: str
    dialect: str
    hasVerifier: bool
    parameters: List[AttrOrTypeParameter]
    traits: List[Trait]
    interfaces: List[str]

    def is_declarative(self, builtins=True, enums=True):
        for param in self.parameters:
            if not param.is_declarative(builtins=builtins, enums=enums):
                return False
        assert len(self.traits) == 0
        for interface in self.interfaces:
            if interface == "::mlir::SubElementTypeInterface::Trait":
                continue
            if interface == "DataLayoutTypeInterface::Trait":
                return False
            print(interface)
            assert False
        return True

    def print(self, indent_level=0):
        print(f"{' ' * indent_level}Type {self.name} {{")

        # Parameters
        print(f"{' ' * (indent_level + indent_size)}Parameters (", end='')
        print(', '.join([f"{param.name}: \"{param.cppType}\"" for param in self.parameters]), end='')
        print(f")")

        # Verifier
        if self.hasVerifier:
            print(f"{' ' * (indent_level + indent_size)}CppVerifier \"verify($_self)\"")

        # TODO traits and interfaces

        # Traits
        print(f"{' ' * indent_level}}}")


@from_json
class Attr:
    name: str
    dialect: str
    hasVerifier: bool
    parameters: List[AttrOrTypeParameter]
    traits: List[Trait]
    interfaces: List[str]

    def is_declarative(self, builtins=True, enums=True):
        for param in self.parameters:
            if not param.is_declarative(builtins=builtins, enums=enums):
                return False
        for interface in self.interfaces:
            if interface == "::mlir::SubElementAttrInterface::Trait":
                continue
            return False
        return True


@dataclass
class Dialect:
    name: str
    ops: Dict[str, Op] = field(default_factory=dict)
    types: Dict[str, Type] = field(default_factory=dict)
    attrs: Dict[str, Attr] = field(default_factory=dict)
    numOperations: int = field(default=0)
    numTypes: int = field(default=0)
    numAttributes: int = field(default=0)

    def add_op(self, op: Op):
        if op.name in self.ops.keys():
            assert "op was already in dialect"
        self.ops[op.name] = op

    def add_type(self, typ: Type):
        if typ.name in self.types.keys():
            assert "type was already in dialect"
        self.types[typ.name] = typ

    def add_attr(self, attr: Attr):
        if attr.name in self.attrs.keys():
            assert "attr was already in dialect"
        self.attrs[attr.name] = attr

    def walk_constraints(self, func: Callable[[Constraint]]):
        for op in self.ops.values():
            op.walk_constraints(func)

    def print(self, indent_level=0):
        print(f"{' ' * indent_level}Dialect {self.name} {{")

        # Types
        for typ in self.types.values():
            typ.print(indent_level + indent_size)

        # TODO Attributes

        # Ops
        for op in self.ops.values():
            op.print(indent_level + indent_size)

        print(f"{' ' * indent_level}}}")


@dataclass
class Stats:
    dialects: Dict[str, Dialect] = field(default_factory=dict)

    def add_op(self, op: Op):
        if op.dialect not in self.dialects:
            self.dialects[op.dialect] = Dialect(op.dialect)
        self.dialects[op.dialect].add_op(op)

    def add_type(self, typ: Type):
        if typ.dialect not in self.dialects:
            self.dialects[typ.dialect] = Dialect(typ.dialect)
        self.dialects[typ.dialect].add_type(typ)

    def add_attr(self, attr: Attr):
        if attr.dialect not in self.dialects:
            self.dialects[attr.dialect] = Dialect(attr.dialect)
        self.dialects[attr.dialect].add_attr(attr)

    def add_stats(self, stats: Stats):
        for op in stats.ops:
            self.add_op(op)
        for typ in stats.types:
            self.add_type(typ)
        for attr in stats.attrs:
            self.add_attr(attr)

    @staticmethod
    def from_json(json):
        stats = Stats()
        for val in json["ops"]:
            stats.add_op(Op.from_json(val))
        for val in json["types"]:
            stats.add_type(Type.from_json(val))
        for val in json["attrs"]:
            stats.add_attr(Attr.from_json(val))
        return stats

    @property
    def ops(self):
        res = []
        for _, dialect in self.dialects.items():
            for _, op in dialect.ops.items():
                res.append(op)
        return res

    @property
    def types(self):
        res = []
        for _, dialect in self.dialects.items():
            for _, typ in dialect.types.items():
                res.append(typ)
        return res

    @property
    def attrs(self):
        res = []
        for _, dialect in self.dialects.items():
            for _, attr in dialect.attrs.items():
                res.append(attr)
        return res

    def walk_constraints(self, func: Callable[[Constraint]]):
        for dialect in self.dialects.values():
            dialect.walk_constraints(func)

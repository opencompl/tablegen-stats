from __future__ import annotations

from copy import copy

from analyze_tablegen.utils import *
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
        m = re.compile(r"::mlir::OpTrait::HasParent<(.*)>::Impl").match(
            self.name)
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
        m = re.compile(
            r"::mlir::OpTrait::SingleBlockImplicitTerminator<(.*)>::Impl"
        ).match(self.name)
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
                r"\(getElementTypeOrSelf\(\$_op.getResult\((.*)\)\) == getElementTypeOrSelf\(\$_op.getOperand\((.*)\)\)\)"
        ).match(self.pred) is not None:
            return True
        if self.pred == "(std::equal_to<>()($tensor.getType().cast<ShapedType>().getElementType(), $result.getType()))":
            return True
        return False


@from_json
class InternalTrait(Trait):
    name: str

    def is_declarative(self) -> bool:
        return False


@dataclass(unsafe_hash=True)
class Constraint(ABC):
    @staticmethod
    def from_predicate(predicate: str) -> Constraint:
        predicate = simplify_expression(predicate)

        m = re.compile(r"\$_self.isa<(.*)>\(\)").match(predicate)
        if m is not None:
            return CppBaseConstraint(predicate[11:-3])

        m = re.compile(r"!\((.*)\)").match(predicate)
        if m is not None:
            constraint = Constraint.from_predicate(m.group(0)[2:-1])
            return NotConstraint(constraint)

        and_operands = separate_on_operator(predicate, "&&")
        if and_operands is not None:
            operand1 = Constraint.from_predicate(and_operands[0])
            operand2 = Constraint.from_predicate(and_operands[1])
            return AndConstraint([operand1, operand2])

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
            return ParametricTypeConstraint(
                "builtin", "integer",
                [IntEqConstraint(int(m.group(1))),
                 AnyConstraint()])

        m = re.match(r"\$_self.isSignlessInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraint(
                    "builtin", "integer",
                    [AnyConstraint(),
                     EnumValueEqConstraint("Signless")])
            return ParametricTypeConstraint(
                "builtin", "integer",
                [IntEqConstraint(int(val)),
                 EnumValueEqConstraint("Signless")])

        m = re.match(r"\$_self.isUnsignedInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraint(
                    "builtin", "integer",
                    [AnyConstraint(),
                     EnumValueEqConstraint("Unsigned")])
            return ParametricTypeConstraint(
                "builtin", "integer",
                [IntEqConstraint(int(val)),
                 EnumValueEqConstraint("Unsigned")])

        m = re.match(r"\$_self.isSignedInteger\((.*)\)", predicate)
        if m is not None:
            val = m.group(1)
            if val == "":
                return ParametricTypeConstraint(
                    "builtin", "integer",
                    [AnyConstraint(),
                     EnumValueEqConstraint("Signed")])
            return ParametricTypeConstraint(
                "builtin", "integer",
                [IntEqConstraint(int(val)),
                 EnumValueEqConstraint("Signed")])

        if predicate == "$_self.isBF16()":
            return ParametricTypeConstraint("builtin", "bf16", [])

        m = re.match(r"\$_self.isF(.*)\(\)", predicate)
        if m is not None:
            return ParametricTypeConstraint("builtin", "f" + m.group(1), [])

        if predicate == "$_self.isIndex()":
            return ParametricTypeConstraint("builtin", "index", [])

        m = re.match(r"\$_self.cast<::mlir::FloatAttr>\(\).getType\(\)(.*)",
                     predicate)
        if m is not None:
            type_predicate = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint(
                "builtin", "float_attr",
                [type_predicate, AnyConstraint()])

        m = re.match(
            r"\$_self.cast<::mlir::ShapedType>\(\).getElementType\(\)(.*)",
            predicate)
        if m is not None:
            element_type_constraint = Constraint.from_predicate("$_self" +
                                                                m.group(1))
            return ShapedTypeConstraint(element_type_constraint)

        m = re.match(r"::mlir::LLVM::getVectorElementType\(\$_self\)(.*)",
                     predicate)
        if m is not None:
            element_type_constraint = Constraint.from_predicate("$_self" +
                                                                m.group(1))
            return LLVMVectorOfConstraint(element_type_constraint)

        m = re.match(
            r"\$_self.cast<::mlir::DenseIntElementsAttr>\(\)( ?).getType\(\)( ?).getElementType\(\)( ?)(.*)",
            predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(4))
            return ParametricTypeConstraint("builtin", "dense",
                                            [sub_constraint])

        m = re.match(
            r"\$_self.cast<::mlir::arm_sve::ScalableVectorType>\(\).getElementType\(\)(.*)",
            predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("dense", "vector",
                                            [AnyConstraint(), sub_constraint])

        m = re.match(
            r"\$_self.cast<::mlir::LLVM::LLVMPointerType>\(\).getElementType\(\)(.*)",
            predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint(
                "llvm", "ptr",
                [sub_constraint, AnyConstraint()])

        m = re.match(
            r"\$_self.cast<::mlir::spirv::CooperativeMatrixNVType>\(\).getElementType\(\)(.*)",
            predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("spv", "coopmatrix", [
                sub_constraint,
                AnyConstraint(),
                AnyConstraint(),
                AnyConstraint()
            ])

        m = re.match(
            r"\$_self.cast<::mlir::pdl::RangeType>\(\).getElementType\(\)(.*)",
            predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("pdl", "range", [sub_constraint])

        m = re.match(
            r"\$_self.cast<::mlir::gpu::MMAMatrixType>\(\).getElementType\(\)(.*)",
            predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("gpu", "mma_matrix", [
                AnyConstraint(),
                AnyConstraint(), sub_constraint,
                AnyConstraint()
            ])

        m = re.match(
            r"\$_self.cast<::mlir::ComplexType>\(\).getElementType\(\)(.*)",
            predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint("builtin", "complex",
                                            [sub_constraint])

        m = re.match(r"\$_self.cast<::mlir::IntegerAttr>\(\).getType\(\)(.*)",
                     predicate)
        if m is not None:
            sub_constraint = Constraint.from_predicate("$_self" + m.group(1))
            return ParametricTypeConstraint(
                "builtin", "integer",
                [sub_constraint, AnyConstraint()])

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

        m = re.match(
            r"\$_self.cast<(::mlir::)?StringAttr>\(\).getValue\(\) == \"(.*)\"",
            predicate)
        if m is not None:
            str_val = m.group(1)
            return ParametricTypeConstraint("builtin", "string",
                                            [StringEqConstraint(str_val)])

        llvm_float_types = [("builtin", "bf16"), ("builtin", "f16"),
                            ("builtin", "f32"), ("builtin", "f64"),
                            ("builtin", "f80"), ("builtin", "f128"),
                            ("llvm", "ppc_fp128")]

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType($_self)":
            return OrConstraint([
                BaseConstraint(dialect, typ)
                for dialect, typ in llvm_float_types
            ])

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType($_self))":
            element_type_constraint = OrConstraint([
                BaseConstraint(dialect, typ)
                for dialect, typ in llvm_float_types
            ])
            return LLVMVectorOfConstraint(element_type_constraint)

        if predicate == "::mlir::LLVM::isCompatibleFloatingPointType($_self.cast<::mlir::LLVM::LLVMPointerType>().getElementType())":
            sub_constraint = OrConstraint([
                BaseConstraint(dialect, typ)
                for dialect, typ in llvm_float_types
            ])
            return ParametricTypeConstraint(
                "llvm", "ptr",
                [sub_constraint, AnyConstraint()])

        if predicate == "::mlir::LLVM::isCompatibleType($_self)":
            return LLVMCompatibleType()

        if predicate == "::mlir::LLVM::isCompatibleType($_self.cast<::mlir::LLVM::LLVMPointerType>().getElementType())":
            return ParametricTypeConstraint(
                "llvm", "ptr",
                [LLVMCompatibleType(), AnyConstraint()])

        if predicate == "::mlir::LLVM::isCompatibleVectorType($_self)":
            llvm_vector_types: List[Constraint] = [
                BaseConstraint("llvm", "fixed_vec"),
                BaseConstraint("llvm", "scalable_vec")
            ]
            vector_elem_float_types = llvm_float_types
            signless_integer = ParametricTypeConstraint(
                "builtin", "integer",
                [AnyConstraint(),
                 EnumValueEqConstraint("Signless")])
            vector_elem_types = OrConstraint([
                BaseConstraint(dialect, typ)
                for dialect, typ in vector_elem_float_types
            ] + [signless_integer])
            vector_type = ParametricTypeConstraint(
                "builtin", "vector",
                [ArrayRefConstraint([AnyConstraint()]), vector_elem_types])
            return OrConstraint(llvm_vector_types + [vector_type])

        if predicate == "$_self.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()":
            return BaseConstraint("builtin", "type_attr")

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
        self.walk_sub_constraints(func)
        func(self)

    @abstractmethod
    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        ...

    @abstractmethod
    def as_str(self, current_irdl_support=False) -> str:
        ...

    def __str__(self):
        return self.as_str()

    def is_variadic(self) -> bool:
        return False


@from_json
class VariadicConstraint(Constraint):
    baseType: Constraint

    def is_declarative(self) -> bool:
        return self.baseType.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.baseType.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(VariadicConstraint(self.baseType.map_constraints(func)))

    def is_variadic(self) -> bool:
        return True

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"Variadic<{self.baseType.as_str(current_irdl_support)}>"


@from_json
class OptionalConstraint(Constraint):
    baseType: Constraint

    def is_declarative(self) -> bool:
        return self.baseType.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.baseType.walk_sub_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(OptionalConstraint(self.baseType.map_constraints(func)))

    def is_variadic(self) -> bool:
        return True

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"Optional<{self.baseType.as_str(current_irdl_support)}>"


@from_json
class TypeDefConstraint(Constraint):
    dialect: str
    name: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        return f"{self.dialect}.{self.name}"


@from_json
class IntegerConstraint(Constraint):
    bitwidth: int

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"IntegerOfSize<{self.bitwidth}>"


@dataclass(unsafe_hash=True)
class AnyConstraint(Constraint):
    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        return f"Any"


@dataclass(unsafe_hash=True)
class BaseConstraint(Constraint):
    dialect: str
    name: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        return f"{self.dialect}.{self.name}"


@dataclass(unsafe_hash=True)
class CppBaseConstraint(Constraint):
    name: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"CppClass<{self.name}>"


@dataclass(unsafe_hash=True)
class ParametricTypeConstraint(Constraint):
    dialect: str
    type: str
    params: List[Constraint]

    def is_declarative(self) -> bool:
        return all(param.is_declarative() for param in self.params)

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        for param in self.params:
            param.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(
            ParametricTypeConstraint(
                self.dialect, self.type,
                [param.map_constraints(func) for param in self.params]))

    def as_str(self, current_irdl_support=False) -> str:
        return f"{self.dialect}.{self.type}<{', '.join([param.as_str(current_irdl_support) for param in self.params])}>"


@dataclass(unsafe_hash=True)
class NotConstraint(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(NotConstraint(self.constraint.map_constraints(func)))

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"Not<{self.constraint.as_str(current_irdl_support)}>"


@dataclass(unsafe_hash=True)
class AndConstraint(Constraint):
    operands: List[Constraint]

    def is_declarative(self) -> bool:
        return all([c.is_declarative() for c in self.operands])

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        for operand in self.operands:
            operand.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(
            AndConstraint(
                [operand.map_constraints(func) for operand in self.operands]))

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"And<{', '.join([operand.as_str(current_irdl_support) for operand in self.operands])}>"


@dataclass(unsafe_hash=True)
class OrConstraint(Constraint):
    operands: List[Constraint]

    def is_declarative(self) -> bool:
        return all([c.is_declarative() for c in self.operands])

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        for operand in self.operands:
            operand.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(
            OrConstraint(
                [operand.map_constraints(func) for operand in self.operands]))

    def as_str(self, current_irdl_support=False) -> str:
        return f"AnyOf<{', '.join([operand.as_str(current_irdl_support) for operand in self.operands])}>"


@dataclass(unsafe_hash=True)
class NotConstraint(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(NotConstraint(self.constraint.map_constraints(func)))

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"Not<{self.constraint.as_str(current_irdl_support)}>"


@dataclass(unsafe_hash=True)
class ShapedTypeConstraint(Constraint):
    elemTypeConstraint: Constraint

    def is_declarative(self) -> bool:
        return self.elemTypeConstraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.elemTypeConstraint.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(
            ShapedTypeConstraint(
                self.elemTypeConstraint.map_constraints(func)))

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"ShapedTypeOf<{self.elemTypeConstraint.as_str(current_irdl_support)}>"


@dataclass(unsafe_hash=True)
class LLVMVectorOfConstraint(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(
            LLVMVectorOfConstraint(self.constraint.map_constraints(func)))

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"LLVMVectorOf<{self.constraint.as_str(current_irdl_support)}>"


@dataclass(unsafe_hash=True)
class AttrArrayOf(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(AttrArrayOf(self.constraint.map_constraints(func)))

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"AttrArrayOf<{self.constraint.as_str(current_irdl_support)}>"


@dataclass(unsafe_hash=True)
class TupleOf(Constraint):
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        self.constraint.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(TupleOf(self.constraint.map_constraints(func)))

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"TupleOf<{self.constraint.as_str(current_irdl_support)}>"


@dataclass(unsafe_hash=True)
class LLVMCompatibleType(Constraint):
    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"LLVMCompatibleType"


@dataclass(unsafe_hash=True)
class EnumValueEqConstraint(Constraint):
    value: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f'{self.value}'


@dataclass(unsafe_hash=True)
class IntEqConstraint(Constraint):
    value: int

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"{self.value}"


@dataclass(unsafe_hash=True)
class StringEqConstraint(Constraint):
    value: str

    def is_declarative(self) -> bool:
        return True

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"{self.value}"


@dataclass(unsafe_hash=True)
class ArrayRefConstraint(Constraint):
    constraints: List[Constraint]

    def is_declarative(self) -> bool:
        return all([c.is_declarative() for c in self.constraints])

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        for constraint in self.constraints:
            constraint.walk_constraints(func)

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(
            ArrayRefConstraint([
                constraint.map_constraints(func)
                for constraint in self.constraints
            ]))

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"[{', '.join([c.as_str(current_irdl_support) for c in self.constraints])}]"


@from_json
class PredicateConstraint(Constraint):
    predicate: str

    @staticmethod
    def from_predicate(predicate: str) -> PredicateConstraint:
        return PredicateConstraint(predicate)

    def is_declarative(self) -> bool:
        self.predicate = simplify_expression(self.predicate)
        m = re.compile(r"\$_self.cast<(.*)>\(\).getRank\(\) == (.*)").match(
            self.predicate)
        if m is not None:
            return True

        if self.predicate == "$_self.cast<::mlir::ShapedType>().hasRank()":
            return True

        if re.compile(
                r"\$_self.cast<::mlir::ArrayAttr>\(\).size\(\) == (.*)").match(
                    self.predicate) is not None:
            return True

        if re.compile(
                r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\)\[0].isa<(.*)>\(\)"
        ).match(self.predicate) is not None:
            return True

        if re.compile(
                r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\)\[1].isSignlessInteger\((.*)\)"
        ).match(self.predicate) is not None:
            return True

        if re.compile("::mlir::spirv::symbolize(.*)").match(self.predicate):
            return True

        if re.compile(
                r"\$_self.cast<mlir::quant::QuantizedType>\(\).getStorageTypeIntegralWidth\(\) == (.*)"
        ).match(self.predicate) is not None:
            return True

        if self.predicate == "$_self.cast<::mlir::TypeAttr>().getValue().isa<::mlir::MemRefType>()":
            return True

        m = re.compile(
            r"\$_self.cast<(.*)>\(\).getNumElements\(\) == (.*)").match(
                self.predicate)
        if m is not None:
            return True

        m = re.compile(r"\$_self.cast<(.*)>\(\).hasStaticShape\(\)").match(
            self.predicate)
        if m is not None:
            return True

        if self.predicate == "$_self.cast<::mlir::ArrayAttr>().size() <= 4":
            return True

        m = re.compile(
            r"\$_self.cast<::mlir::LLVM::LLVMStructType>\(\).getBody\(\).size\(\) == (.*)"
        ).match(self.predicate)
        if m is not None:
            return True

        # Harder cases:
        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getInt() >= 0":
            return False

        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getInt() <= 3":
            return False

        if self.predicate == "$_self.cast<IntegerAttr>().getValue().isStrictlyPositive()":
            return False

        if self.predicate == "$_self.cast<::mlir::IntegerAttr>().getValue().isNegative()":
            return False

        if self.predicate == "isStrided($_self.cast<::mlir::MemRefType>())":
            return False

        if self.predicate == "$_self.cast<::mlir::LLVM::LLVMStructType>().isOpaque()":
            return False

        print(self.predicate)
        assert False

    def walk_sub_constraints(self, func: Callable[[Constraint]]):
        pass

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> Constraint:
        return func(self)

    def as_str(self, current_irdl_support=False) -> str:
        if current_irdl_support:
            return "Any"
        return f"CPPPredicate<\"{self.predicate}\">"


@from_json
class NamedConstraint:
    name: str
    constraint: Constraint

    def is_declarative(self) -> bool:
        return self.constraint.is_declarative()

    def map_constraints(
            self, func: Callable[[Constraint], Constraint]) -> NamedConstraint:
        return NamedConstraint(self.name,
                               self.constraint.map_constraints(func))

    def as_str(self, current_irdl_support=False) -> str:
        name = self.name if len(self.name) != 0 else "__empty__"
        return f"{name}: {self.constraint.as_str(current_irdl_support)}"


@dataclass
class NamedRegion:
    name: str
    isVariadic: bool
    isSingleBlock: bool
    hasTerminator: Optional[str] = field(default=None)

    @staticmethod
    def from_json(json):
        name = json["name"]
        is_variadic = json["isVariadic"]
        constraint = simplify_expression(json["constraint"]["predicate"])
        if constraint == "::llvm::hasNItems($_self, 1)":
            is_single_block = True
        else:
            is_single_block = False
            assert constraint == "true"
        return NamedRegion(name, is_variadic, is_single_block)


@from_json
class Op:
    name: str
    dialect: str
    hasVerifier: bool
    numOperands: int
    numVariableLengthOperands: int
    numResults: int
    hasNoVariadicRegions: bool
    numSuccessors: int
    hasAssemblyFormat: bool
    operands: List[NamedConstraint]
    results: List[NamedConstraint]
    regions: List[NamedRegion]
    attributes: Dict[str, Constraint]
    traits: List[Trait]
    interfaces: List[str]

    def __post_init__(self):
        if self.name.startswith(self.dialect + "."):
            self.name = self.name[len(self.dialect + "."):]
        for trait in self.traits:
            if not isinstance(trait, NativeTrait):
                continue
            m = re.match(
                r"::mlir::OpTrait::SingleBlockImplicitTerminator<(.*)>::Impl",
                trait.name)
            if m is not None:
                for region in self.regions:
                    region.isSingleBlock = True
                    region.hasTerminator = m.group(1)
            if trait.name == "::mlir::OpTrait::SingleBlock":
                for region in self.regions:
                    region.isSingleBlock = True

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

    def is_declarative(self,
                       check_traits: bool = True,
                       check_interfaces: bool = True) -> bool:
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

    def map_constraints(self, func: Callable[[Constraint], Constraint]) -> Op:
        new_op = copy(self)
        new_op.operands = [
            operand.map_constraints(func) for operand in new_op.operands
        ]
        new_op.results = [
            result.map_constraints(func) for result in new_op.results
        ]
        new_op.attributes = {
            name: attr.map_constraints(func)
            for name, attr in new_op.attributes.items()
        }
        return new_op

    def as_str(self, indent_level=0, current_irdl_support=False):
        res = ""
        name = self.name if self.name[0].isalpha() or self.name[0] == "_" else '"' + self.name + '"'
        res += f"{' ' * indent_level}irdl.operation {name}"
        if len(self.operands) == 0 and len(self.results) == 0:
            return res
        res += " {\n"

        # Operands
        if len(self.operands) != 0:
            res += f"{' ' * (indent_level + indent_size)}irdl.operands("
            res += f",\n{' ' * (indent_level + indent_size + len('irdl.operands ('))}".join([operand.as_str(current_irdl_support) for operand in self.operands])
            res += ")\n"

        # Results
        if len(self.results) != 0:
            res += f"{' ' * (indent_level + indent_size)}irdl.results("
            res += f",\n{' ' * (indent_level + indent_size + len('irdl.results ('))}".join([result.as_str(current_irdl_support) for result in self.results])
            res += ")\n"

        # TODO Attributes

        res += f"{' ' * indent_level}}}\n"
        return res

    def __str__(self):
        return self.as_arg()


@from_json
class AttrOrTypeParameter:
    name: str
    cppType: str

    def get_group(self):
        base_names = [
            "Type", "::mlir::Type", "Attribute", "ShapedType",
            "DenseElementsAttr", "DenseIntElementsAttr", "StringAttr",
            "VerCapExtAttr", "DictionaryAttr"
        ]
        if self.cppType in base_names:
            return "attr/type"

        base_array_names = [
            "::llvm::ArrayRef<Attribute>", "::llvm::ArrayRef<NamedAttribute>",
            "ArrayRef<Type>", "::llvm::ArrayRef<FlatSymbolRefAttr>"
        ]
        if self.cppType in base_array_names:
            return "attr/type array"

        integer_names = [
            "unsigned", "uintptr_t", "int64_t", "uint32_t", "int", "APInt",
            "bool"
        ]
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

        enum_names = [
            "Scope", "Dim", "ImageDepthInfo", "ImageArrayedInfo",
            "ImageSamplingInfo", "ImageSamplerUseInfo", "ImageFormat",
            "StorageClass", "Optional<StorageClass>", "Version", "Capability",
            "Extension", "Vendor", "DeviceType", "CombiningKind",
            "SignednessSemantics", "FastmathFlags", "linkage::Linkage"
        ]
        if self.cppType in enum_names:
            return "enum"

        enum_array_names = [
            "ArrayRef<OffsetInfo>",
            "::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>>",
            "ArrayRef<MemberDecorationInfo>",
            "::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType>",
            "ArrayRef<Capability>", "ArrayRef<Extension>"
        ]
        if self.cppType in enum_array_names:
            return "enum array"

        affine_names = ["AffineMap", "IntegerSet"]
        if self.cppType in affine_names:
            return "affine"

        affine_array_names = ["::llvm::ArrayRef<AffineMap>"]
        if self.cppType in affine_array_names:
            return "affine array"

        if self.cppType == "Location":
            return "location"

        if self.cppType == "::llvm::ArrayRef<Location>":
            return "location array"

        if self.cppType == "TypeID":
            return "type id"

        if self.cppType == "LLVMStruct":
            return "llvm"

        raise Exception(f"Unexpected attr or type parameter: {self.cppType}")

    def is_declarative(self, builtins=True, enums=True):
        assert (not enums or builtins)
        base_names = [
            "Type", "::mlir::Type", "Attribute", "ShapedType",
            "DenseElementsAttr", "DenseIntElementsAttr", "StringAttr",
            "VerCapExtAttr", "DictionaryAttr"
        ]
        if self.cppType in base_names:
            return True

        # integers and floats
        builtin_names = [
            "unsigned", "uintptr_t", "int64_t", "uint32_t", "int", "APInt",
            "::llvm::APFloat", "bool", "double", "Identifier",
            "::llvm::StringRef"
        ]
        if self.cppType in builtin_names:
            return builtins

        # arrays
        arrays = [
            "::llvm::ArrayRef<int64_t>", "::llvm::ArrayRef<Attribute>",
            "::llvm::ArrayRef<NamedAttribute>", "ArrayRef<Type>",
            "ArrayRef<char>", "ArrayRef<StringRef>",
            "::llvm::ArrayRef<FlatSymbolRefAttr>", "ArrayRef<double>",
            "ArrayRef<int64_t>"
        ]
        if self.cppType in arrays:
            return builtins

        # Enums
        enum_names = [
            "Scope", "Dim", "ImageDepthInfo", "ImageArrayedInfo",
            "ImageSamplingInfo", "ImageSamplerUseInfo", "ImageFormat",
            "StorageClass", "Optional<StorageClass>", "ArrayRef<OffsetInfo>",
            "::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>>",
            "ArrayRef<MemberDecorationInfo>",
            "::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType>",
            "Version", "Capability", "ArrayRef<Capability>", "Extension",
            "ArrayRef<Extension>", "Vendor", "DeviceType", "CombiningKind",
            "SignednessSemantics", "FastmathFlags", "linkage::Linkage"
        ]
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
    cppName: str
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

    def as_str(self, *, indent_level=0, current_irdl_support=False) -> str:
        res = ""
        res += f"{' ' * indent_level}irdl.type {self.name} {{\n"

        # Parameters
        res += f"{' ' * (indent_level + indent_size)}irdl.parameters("
        res += ', '.join([
            f"{param.name}: {'Any' if current_irdl_support else param.cppType}" for param in self.parameters
        ])
        res += f")\n"

        # TODO verifiers, traits and interfaces

        # Traits
        res += f"{' ' * indent_level}}}"
        return res

    def __str__(self):
        return self.as_str()


@from_json
class Attr:
    name: str
    dialect: str
    cppName: str
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

    def as_str(self, *, indent_level=0, current_irdl_support=False) -> str:
        res = ""
        res += f"{' ' * indent_level}irdl.attribute {self.name} {{\n"

        # Parameters
        res += f"{' ' * (indent_level + indent_size)}irdl.parameters("
        res += ', '.join([
            f"{param.name}: {'Any' if current_irdl_support else param.cppType}" for param in self.parameters
        ])
        res += f")\n"

        # TODO verifiers, traits and interfaces

        # Traits
        res += f"{' ' * indent_level}}}"
        return res


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

    def map_constraints(self, func: Callable[[Constraint],
                                             Constraint]) -> Dialect:
        new_dialect = copy(self)
        new_dialect.ops = {
            name: op.map_constraints(func)
            for name, op in new_dialect.ops.items()
        }
        return new_dialect

    def as_str(self, *, indent_level=0, current_irdl_support=False) -> str:
        res = ""
        res += f"{' ' * indent_level}irdl.dialect {self.name} {{\n"

        # Types
        for typ in self.types.values():
            res += typ.as_str(indent_level=indent_level + indent_size, current_irdl_support=current_irdl_support) + "\n"

        if not current_irdl_support:
            for attr in self.attrs.values():
                res += attr.as_str(indent_level=indent_level + indent_size,
                                  current_irdl_support=current_irdl_support) + "\n"

        # Ops
        for op in self.ops.values():
            res += op.as_str(indent_level + indent_size, current_irdl_support=current_irdl_support) + "\n"

        res += f"{' ' * indent_level}}}\n"
        return res

    def __str__(self):
        return self.as_str()


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

    def map_constraints(self, func: Callable[[Constraint],
                                             Constraint]) -> Stats:
        return Stats({
            name: dialect.map_constraints(func)
            for name, dialect in self.dialects.items()
        })

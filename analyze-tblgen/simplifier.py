from irdl import *
import typing


class Simplifier(ABC):

    @staticmethod
    @abstractmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        ...


class MergeOrAnd(Simplifier):

    @staticmethod
    def should_merge(typ: Union[typing.Type[AndConstraint], typing.Type[OrConstraint]],
                     constraint: Union[AndConstraint, OrConstraint]) -> bool:
        for operand in constraint.operands:
            if isinstance(operand, typ):
                return True
        return False

    @staticmethod
    def merge(typ: Union[typing.Type[AndConstraint], typing.Type[OrConstraint]],
              constraint: Union[AndConstraint, OrConstraint]) -> Constraint:
        new_operands = []
        for operand in constraint.operands:
            if isinstance(operand, typ):
                for sub_operand in operand.operands:
                    new_operands.append(sub_operand)
            else:
                new_operands.append(operand)
        # noinspection PyArgumentList
        return typ(new_operands)

    @staticmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        def simplify_(typ: Union[typing.Type[AndConstraint], typing.Type[OrConstraint]]) -> Optional[Constraint]:
            if isinstance(constraint, typ):
                if not MergeOrAnd.should_merge(typ, constraint):
                    return None
                return MergeOrAnd.merge(typ, constraint)
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
            if isinstance(other, ParametricTypeConstraint) and other.base == constraint.name:
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
                if AndRedundantSimplifier.is_redundant(sub_constraint, sub_constraint2):
                    to_remove.add(idx)
                    continue

        if len(to_remove) == 0:
            return None

        new_operands = []
        for idx, operand in enumerate(constraint.operands):
            if idx not in to_remove:
                new_operands.append(operand)
        return AndConstraint(new_operands)


class TrivialAndSimplifier(Simplifier):
    @staticmethod
    def simplify(constraint: Constraint) -> Optional[Constraint]:
        if isinstance(constraint, AndConstraint):
            if len(constraint.operands) == 1:
                return constraint.operands[0]
        return None


def simplify(stats: Stats) -> Stats:
    simplifiers = [MergeOrAnd, AndRedundantSimplifier, TrivialAndSimplifier]

    def map_constraint(constraint: Constraint) -> Constraint:
        for simplifier in simplifiers:
            new_constraint = simplifier.simplify(constraint)
            if new_constraint is not None:
                constraint = new_constraint
        return constraint

    return stats.map_constraints(map_constraint)

from __future__ import annotations

from dataclasses import dataclass
from typing import *
import re


def has_balanced_parentheses(val: str) -> bool:
    paren_level = 0
    for char in val:
        if char == '(':
            paren_level += 1
        if char == ')':
            paren_level -= 1
            if paren_level < 0:
                return False
    return paren_level == 0


def remove_outer_parentheses(val: str) -> str:
    val = val.strip()
    assert has_balanced_parentheses(val)
    if val[0] != '(' or val[-1] != ')':
        return val
    if has_balanced_parentheses(val[1:-1]):
        return remove_outer_parentheses(val[1:-1])
    return val


def simplify_expression(val: str) -> str:
    val = remove_outer_parentheses(val)
    val = re.sub(" +", " ", val)
    return val


def separate_on_operator(val: str, operator: str) -> Optional[Tuple[str, str]]:
    val = remove_outer_parentheses(val)
    paren_level = 0
    for idx, char in enumerate(val):
        if char == '(':
            paren_level += 1
        if char == ')':
            paren_level -= 1
            assert paren_level >= 0
        if paren_level == 0:
            if val[idx:idx + len(operator)] == operator:
                return val[0:idx], val[idx + len(operator):]
    return None


def _from_json(json, typ: Type):
    if get_origin(typ) == list:
        arg_typ = get_args(typ)[0]
        res = []
        for val in json:
            res.append(_from_json(val, arg_typ))
        return res
    if get_origin(typ) == dict:
        args = get_args(typ)
        assert args[0] == str
        arg_typ = args[1]
        res = dict()
        for key, val in json.items():
            res[key] = _from_json(val, arg_typ)
        return res
    assert get_origin(typ) is None
    if isinstance(json, typ):
        return json
    return typ.from_json(json)


def from_json(cls):
    @dataclass(eq=True, unsafe_hash=True)
    class FromJsonWrapper(dataclass(cls)):
        def __repr__(self):
            return cls.__name__[:-5] + "(" + ", ".join([f"{key}={self.__dict__[key]}" for key in
                                                        cls.__dataclass_fields__.keys()]) + ")"

        @staticmethod
        def from_json(json):
            arg_dict = dict()
            for name, typ in get_type_hints(cls).items():
                arg_dict[name] = _from_json(json[name], typ)
            return FromJsonWrapper(**arg_dict)

    return FromJsonWrapper

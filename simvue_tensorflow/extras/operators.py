"""Common.

Classses which could be used in the construction of multiple adapters.
"""
# ruff: noqa: DOC201

import enum
from typing import Callable, Literal, Optional, Union

from pydantic import BaseModel, Field, PositiveInt, ValidationInfo, field_validator

NAME_REGEX: str = r"^[a-zA-Z0-9\-\_\s\/\.:]+$"


# temp
class Operator(str, enum.Enum):
    """The operator to use to compare the reduced evaluation value to a given target threshold."""

    MORE_THAN = ">"
    LESS_THAN = "<"
    MORE_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="


OPERATORS: dict[str, Callable[[Union[int, float], Union[int, float]], bool]] = {
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    ">=": lambda x, y: x >= y,
    "<=": lambda x, y: x <= y,
    "==": lambda x, y: x == y,
}

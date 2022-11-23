from copy import deepcopy
from typing import Any, List
from typing_extensions import Self
from .variable_base import PointBase

__all__ = ["Memory"]


class Memory:
    _format_string = "{}|{}"

    def __init__(self, dataset: List[PointBase]):
        self._raw_dataset = dataset
        self._memory = {}

    def __len__(self) -> int:
        return len(self._raw_dataset)

    @property
    def raw_dataset(self) -> List[PointBase]:
        return self._raw_dataset

    def save(self, data: Any, p_idx: int, given_p_idx: int):
        key_name: str = self._format_string.format(p_idx, given_p_idx)
        assert (
            key_name not in self._memory
        ), f"already used {key_name} as {self._memory[key_name]}"
        self._memory[key_name] = data

    def get(self, p_idx, given_p_idx) -> Any:
        return self._memory[self._format_string.format(p_idx, given_p_idx)]

    def clone(self) -> Self:
        return deepcopy(self)

    def __repr__(self) -> str:
        return str(self._memory)

    def __str__(self) -> str:
        return str(self._memory)

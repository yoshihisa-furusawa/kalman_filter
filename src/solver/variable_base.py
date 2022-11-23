from typing import Any, Dict, List, Protocol
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray

__all__ = ["ObservedDataBase", "SystemDataBase", "PointBase", "NP_FLOAT", "MemoryBase"]

NP_FLOAT = NDArray[np.float32]


class ObservedDataBase(Protocol):
    H: NP_FLOAT
    R: NP_FLOAT
    observed_y: NP_FLOAT


class SystemDataBase(Protocol):
    F: NP_FLOAT
    G: NP_FLOAT
    Q: NP_FLOAT

    def predicted_data(self) -> Dict[str, NP_FLOAT]:
        pass

    def filtered_data(self) -> Dict[str, NP_FLOAT]:
        pass


class PointBase(Protocol):
    observe: ObservedDataBase
    system: SystemDataBase

    def __call__(
        self, p_idx: int, data: Dict[str, Any], memory: "MemoryBase"
    ) -> Dict[str, Any]:
        pass


class MemoryBase(Protocol):
    _format_string: str = "{}|{}"
    _memory: Dict[str, Any]
    raw_dataset: List[PointBase]

    def __init__(self, dataset: List[PointBase]):
        pass

    def __len__(self) -> int:
        return len(self._raw_dataset)

    def save(self, data: Any, p_idx: int, given_p_idx: int):
        pass

    def get(self, p_idx, given_p_idx) -> Any:
        pass

    def clone(self) -> Self:
        pass

    def __repr__(self) -> str:
        return str(self._memory)

    def __str__(self) -> str:
        return str(self._memory)

from typing import Any, Dict, List
from .variable_base import MemoryBase, PointBase


class StandardSolverField:
    def __init__(
        self,
        memory: MemoryBase,
    ):
        assert isinstance(memory, MemoryBase)
        self._memory = memory
        self._point_list: List[PointBase] = memory.raw_dataset

    @property
    def memory(self):
        return self._memory

    def __call__(self, init_system_data: Dict[str, Any]):
        # NOTE: 初期値の保存
        self.memory.save(init_system_data, 0, 0)

        filtered_result = []
        for p_idx, each_point in enumerate(self._point_list):
            if p_idx == 0:
                # NOTE: In first time, the model predicts (t_1_given_0) from initial condition as filtered data (t_0_given_0).
                system_data = init_system_data

            system_data = each_point(
                p_idx=p_idx,
                data=system_data,
                memory=self._memory,
            )
            filtered_result.append(system_data)
        return filtered_result

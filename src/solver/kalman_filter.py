from dataclasses import dataclass
from typing import TypedDict, Union
import numpy as np
from .variable_base import NP_FLOAT
from .memory import Memory

__all__ = ["KFObservedData", "KFSystemData", "KalmanFilterPoint", "rts_smoother"]


def check_numpy(*arrays):
    for array in arrays:
        assert isinstance(array, np.ndarray)


def check_square_matrix(x):
    assert x.ndim == 2
    x1, x2 = x.shape
    assert x1 == x2


def check_ndim(x: np.ndarray, ndim: int):
    assert x.ndim == ndim


class KFOutput(TypedDict):
    X: NP_FLOAT
    V: NP_FLOAT


@dataclass
class KFObservedData:
    """
    NOTE:
        y_{t} = H @ x_{t} + w_{t}
        w_{t} ~ Norm(0, R)
    Attributes:
        H: weight for state vector x_{t} (in R^{k}), which shape is R^{l \times k}.
        R: convariance matirx in normal distribution, which shape is R^{l \times l}.
        observed_y: observed data point, which type is vector (in R^{l}).
    """

    __slots__ = ("H", "R", "observed_y")
    H: NP_FLOAT
    R: NP_FLOAT
    observed_y: NP_FLOAT

    def __post_init__(self):
        check_numpy(self.H, self.R, self.observed_y)
        check_ndim(self.H, ndim=2)
        check_ndim(self.observed_y, ndim=1)

        l_in_H, _ = self.H.shape

        check_square_matrix(self.R)
        l1_in_R, _ = self.R.shape

        assert l_in_H == l1_in_R
        assert l_in_H == len(self.observed_y)


@dataclass
class KFSystemData:
    """
    NOTE:
        x_{t} = F @ x_{t-1} + G_{t} @ v_{t}
        v_{t} ~ Norm(0, Q)
    Attributes:
        F: weight for state vector x_{t-1} (in R^{k}), which shape is R^{k \times k}.
        G: weight for noise vector v_{t} (in R^{m}), which shape is R^{k \times m}.
        Q: convariance matirx in normal distribution, which shape is R^{m \times m}.
    """

    __slots__ = ("F", "G", "Q")
    F: NP_FLOAT
    G: NP_FLOAT
    Q: NP_FLOAT

    def __post_init__(self):
        check_numpy(self.F, self.G, self.Q)

        check_ndim(self.F, ndim=2)
        check_ndim(self.G, ndim=2)
        check_ndim(self.Q, ndim=2)

        check_square_matrix(self.F)
        k_in_F, _ = self.F.shape

        check_square_matrix(self.Q)
        m_in_Q, _ = self.Q.shape

        k_in_G, m_in_G = self.G.shape

        assert k_in_F == k_in_G
        assert m_in_G == m_in_Q

    def predicted_data(self, X: NP_FLOAT, V: NP_FLOAT) -> KFOutput:
        return {
            "X": self.F @ X,
            "V": self.F @ V @ self.F.T + self.G @ self.Q @ self.G.T,
        }

    def filtered_data(
        self,
        X: NP_FLOAT,
        V: NP_FLOAT,
        observed_data: KFObservedData,
    ) -> KFOutput:
        kalman_gain = calc_kalman_gain(V, observed_data.H, observed_data.R)
        filtered_X = X + kalman_gain @ (observed_data.observed_y - observed_data.H @ X)
        filtered_V = V - kalman_gain @ observed_data.H @ V
        return {"X": filtered_X, "V": filtered_V}


def calc_kalman_gain(predicted_V: NP_FLOAT, H: NP_FLOAT, R: NP_FLOAT) -> NP_FLOAT:
    return predicted_V @ H.T @ np.linalg.inv(H @ predicted_V @ H.T + R)


@dataclass
class KalmanFilterPoint:
    observe: KFObservedData
    system: KFSystemData

    def __call__(self, p_idx: int, data: KFOutput, memory: Memory) -> KFOutput:
        kf_output: KFOutput = self.system.predicted_data(**data)
        memory.save(kf_output, p_idx + 1, p_idx)

        kf_output: KFOutput = self.system.filtered_data(
            **kf_output, observed_data=self.observe
        )
        memory.save(kf_output, p_idx + 1, p_idx + 1)

        return kf_output


def rts_smoother(memory: Memory, get_memory: bool = False) -> Union[Memory, KFOutput]:
    memory = memory.clone()
    time_length = len(memory)

    for idx, data_t in enumerate(reversed(memory.raw_dataset), start=1):
        # p_idx: int
        # data: KalmanFilterPoint
        p_idx: int = time_length - idx
        t_given_t: KFOutput = memory.get(p_idx, p_idx)
        t_add_1_given_T: KFOutput = memory.get(p_idx + 1, time_length)
        t_add_1_given_t: KFOutput = memory.get(p_idx + 1, p_idx)

        A_t = t_given_t["V"] @ data_t.system.F.T @ np.linalg.inv(t_add_1_given_t["V"])
        X_t_given_T = t_given_t["X"] + A_t @ (
            t_add_1_given_T["X"] - t_add_1_given_t["X"]
        )
        V_t_given_T = (
            t_given_t["V"] + A_t @ (t_add_1_given_T["V"] - t_add_1_given_t["V"]) @ A_t.T
        )
        memory.save({"X": X_t_given_T, "V": V_t_given_T}, p_idx, time_length)

    if get_memory:
        return memory

    X_list, V_list = [], []
    for t in range(1, time_length + 1):
        t_given_T = memory.get(t, time_length)
        X_list.append(t_given_T["X"])
        V_list.append(t_given_T["V"])
    return {"X": np.concatenate(X_list), "V": np.concatenate(V_list)}

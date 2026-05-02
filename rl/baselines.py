import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
baselines.py — Các chiến lược baseline để so sánh với DQN Agent.
Tất cả đều implement cùng interface: get_action(state_vector, context) -> int
  action = 0: Edge
  action = 1: Cloud

State vector (8-dim):
  [critical_count, is_complex, edge_lat, cloud_lat,
   edge_cpu, cloud_cpu, edge_pending_norm, cloud_pending_norm]
"""
import itertools


class RoundRobinBaseline:
    """Luân phiên Edge → Cloud → Edge → ... không quan tâm gì cả."""
    def __init__(self):
        self._cycle = itertools.cycle([0, 1])
        self.name = "RoundRobin"

    def get_action(self, state_vector=None, context=None) -> int:
        return next(self._cycle)


class AlwaysCloudBaseline:
    """Luôn lên Cloud — chi phí cao nhất nhưng chất lượng tốt nhất."""
    name = "AlwaysCloud"

    def get_action(self, state_vector=None, context=None) -> int:
        return 1


class AlwaysEdgeBaseline:
    """Luôn về Edge — chi phí thấp nhưng chất lượng không ổn định."""
    name = "AlwaysEdge"

    def get_action(self, state_vector=None, context=None) -> int:
        return 0


class LatencyBasedBaseline:
    """
    Greedy dựa trên latency lịch sử:
    - Nếu câu phức tạp (is_complex) → Cloud (an toàn y tế)
    - Nếu history_latency Edge < Cloud - threshold → Edge
    - Ngược lại → Cloud
    """
    def __init__(self, latency_threshold: float = 2.0):
        self.threshold = latency_threshold
        self.name = "LatencyBased"

    def get_action(self, state_vector, context=None) -> int:
        is_complex = bool(state_vector[1])
        edge_lat   = state_vector[2]
        cloud_lat  = state_vector[3]

        if is_complex:
            return 1
        if edge_lat <= cloud_lat - self.threshold:
            return 0
        return 1


class CpuAwareBaseline:
    """
    Greedy dựa trên CPU load:
    - Nếu câu phức tạp → Cloud
    - Nếu edge_cpu < cpu_threshold → Edge
    - Ngược lại → Cloud
    """
    def __init__(self, cpu_threshold: float = 0.75):
        self.threshold = cpu_threshold
        self.name = "CpuAware"

    def get_action(self, state_vector, context=None) -> int:
        is_complex = bool(state_vector[1])
        edge_cpu   = state_vector[4]

        if is_complex:
            return 1
        if edge_cpu < self.threshold:
            return 0
        return 1


class LoadBalancerBaseline:
    """
    Dựa trên pending requests — baseline mới tận dụng state 8-dim.
    Route đến node có ít pending request hơn, trừ khi câu phức tạp.
    """
    name = "LoadBalancer"

    def get_action(self, state_vector, context=None) -> int:
        is_complex = bool(state_vector[1])
        edge_pending = state_vector[6]
        cloud_pending = state_vector[7]

        if is_complex:
            return 1
        return 0 if edge_pending <= cloud_pending else 1


def get_all_baselines() -> list:
    return [
        RoundRobinBaseline(),
        AlwaysCloudBaseline(),
        AlwaysEdgeBaseline(),
        LatencyBasedBaseline(),
        CpuAwareBaseline(),
        LoadBalancerBaseline(),
    ]

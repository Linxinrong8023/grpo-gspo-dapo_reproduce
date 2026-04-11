"""
algorithms/__init__.py - 算法包公开接口。

使用方式：
    from src.post_training_contrast.algorithms import resolve_algorithm

    algo = resolve_algorithm("grpo")   # → GRPOAlgorithm 实例
    algo = resolve_algorithm("dapo")   # → DAPOAlgorithm 实例
"""

from src.post_training_contrast.algorithms.base import (
    Algorithm,
    PolicyLossBatch,
    PolicyLossOutput,
    compute_group_relative_advantages,
)
from src.post_training_contrast.algorithms.dapo import DAPOAlgorithm
from src.post_training_contrast.algorithms.grpo import GRPOAlgorithm
from src.post_training_contrast.algorithms.gspo import GSPOAlgorithm

__all__ = [
    "Algorithm",
    "DAPOAlgorithm",
    "GRPOAlgorithm",
    "GSPOAlgorithm",
    "PolicyLossBatch",
    "PolicyLossOutput",
    "compute_group_relative_advantages",
    "resolve_algorithm",
]

_ALGORITHM_REGISTRY: dict[str, type[Algorithm]] = {
    "grpo": GRPOAlgorithm,
    "gspo": GSPOAlgorithm,
    "dapo": DAPOAlgorithm,
}


def resolve_algorithm(name: str) -> Algorithm:
    """按名称创建算法实例。支持 'grpo' / 'gspo' / 'dapo'（大小写不敏感）。"""
    key = name.lower()
    if key not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"暂不支持的算法：{name!r}。可用算法：{list(_ALGORITHM_REGISTRY)}"
        )
    return _ALGORITHM_REGISTRY[key]()

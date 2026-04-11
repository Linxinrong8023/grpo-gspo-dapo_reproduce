"""
visualization.py - 训练和评测结果可视化工具。

提供三个函数：
  - plot_training_curves       训练曲线（loss, accuracy, truncation_rate 等）
  - plot_eval_summary          评测结果（总准确率 + 切片维度柱状图）
  - plot_algorithm_comparison  多算法准确率对比柱状图
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 必须在 pyplot import 前设置，兼容无显示器的服务器环境
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False
try:
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
except Exception:
    pass

_COLORS = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#3B82F6", "#EC4899"]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_fig(fig: plt.Figure, save_path: str | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"图片已保存: {save_path}")


# ── 训练曲线 ─────────────────────────────────────────────────────


def plot_training_curves(
    summary_path: str,
    metrics: list[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """读取 train_summary.json，把指定指标画成随 step 变化的折线图。

    Parameters
    ----------
    summary_path : str
        训练总结 JSON 路径，如 outputs/train/grpo_math/train_summary.json
    metrics : list[str] | None
        要画的指标名列表。默认画 accuracy, truncation_rate, parse_fail_rate,
        mean_response_length 和 update_losses（取均值）。
    """
    data = _load_json(summary_path)
    steps_data = data.get("steps", [])
    if not steps_data:
        print("没有找到训练步骤数据。")
        return

    if metrics is None:
        metrics = ["group_signal_rate", "group_mixed_answer_rate", "format_reward_mean", "accuracy", "parse_fail_rate"]

    has_loss = any("update_losses" in s and s["update_losses"] for s in steps_data)
    if has_loss:
        metrics = ["mean_loss"] + metrics

    algo_name = data.get("trainer_config", {}).get("algorithm_name", "RL")
    model_name = Path(data.get("model", "unknown")).name

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    fig.suptitle(f"Training Curves: {algo_name.upper()} — {model_name}", fontsize=14, fontweight="bold")

    x = [s["step"] for s in steps_data]
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        color = _COLORS[idx % len(_COLORS)]

        if metric == "mean_loss":
            y = [
                (sum(s["update_losses"]) / len(s["update_losses"]) if s.get("update_losses") else 0.0)
                for s in steps_data
            ]
            label = "Mean Loss (per step)"
        else:
            y = [s.get(metric, 0.0) for s in steps_data]
            label = metric.replace("_", " ").title()

        ax.plot(x, y, marker="o", markersize=4, color=color, linewidth=1.5, label=label)
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    _save_fig(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


# ── 评测结果 ─────────────────────────────────────────────────────


def plot_eval_summary(
    result_path: str,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """读取评测 JSON，将不同维度分别画成独立的图片。"""
    data = _load_json(result_path)
    
    # 兼容 evaluator.py 的输出格式
    slice_metrics = data.get("slice_metrics", {})
    if not slice_metrics:
        if "by_level" in data:
            slice_metrics["Level"] = data["by_level"]
        if "by_subject" in data:
            slice_metrics["Subject"] = data["by_subject"]
            
    dataset_name = data.get("dataset_name", "unknown")
    accuracy = data.get("accuracy", 0.0)
    num_samples = data.get("num_samples", 0)

    # 路径后缀处理
    base_dir = Path(save_path).parent if save_path else Path("")
    base_name = Path(save_path).stem if save_path else "eval_summary"

    def get_save_path(suffix: str) -> str | None:
        if not save_path:
            return None
        return str(base_dir / f"{base_name}_{suffix}.png")

    # ── 1. 第一张独立的图：总评 ──
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["Overall"], [accuracy], color=_COLORS[0], width=0.4, edgecolor="none")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Overall Accuracy: {dataset_name}\n(n={num_samples})", fontsize=14, fontweight="bold", pad=15)
    ax.bar_label(ax.containers[0], fmt="%.2f", padding=3, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    _save_fig(fig, get_save_path("overall"))
    if show:
        plt.show()
    plt.close(fig)

    # ── 2. 后续每张维度独立画一张图 ──
    for idx, (slice_name, categories) in enumerate(slice_metrics.items()):
        labels = list(categories.keys())
        # 根据横坐标类别数量动态决定画板宽度
        width = max(6, len(labels) * 1.2)
        fig, ax = plt.subplots(figsize=(width, 5))
        
        accs = [categories[k]["accuracy"] for k in labels]
        totals = [categories[k]["total"] for k in labels]
        bar_colors = [_COLORS[i % len(_COLORS)] for i in range(len(labels))]
        
        bars = ax.bar(labels, accs, color=bar_colors, width=0.5, edgecolor="none")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy By {slice_name}: {dataset_name}", fontsize=14, fontweight="bold", pad=15)
        
        # 类目多时自动倾斜，少时平放
        ax.tick_params(axis="x", rotation=15 if len(labels) > 3 else 0, labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        for bar, acc, total in zip(bars, accs, totals):
            correct = int(round(acc * total))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{correct}/{total}\n({acc:.1%})",
                ha="center", va="bottom", fontsize=10,
            )

        plt.tight_layout()
        _save_fig(fig, get_save_path(slice_name.lower()))
        if show:
            plt.show()
        plt.close(fig)


# ── 多算法对比 ───────────────────────────────────────────────────


def plot_algorithm_comparison(
    result_paths: dict[str, str],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """对比多个算法在同一数据集上的最终准确率。

    Parameters
    ----------
    result_paths : dict[str, str]
        算法名 → 评测 JSON 路径的映射。
        例如: {"GRPO": "outputs/eval/grpo.json", "DAPO": "outputs/eval/dapo.json"}
    """
    algo_names, accuracies, ns = [], [], []
    for algo_name, path in result_paths.items():
        d = _load_json(path)
        algo_names.append(algo_name)
        accuracies.append(d.get("accuracy", 0.0))
        ns.append(d.get("num_samples", 0))

    dataset_name = _load_json(list(result_paths.values())[0]).get("dataset_name", "")
    bar_colors = [_COLORS[i % len(_COLORS)] for i in range(len(algo_names))]

    fig, ax = plt.subplots(figsize=(max(6, len(algo_names) * 2), 5))
    bars = ax.bar(algo_names, accuracies, color=bar_colors, width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Algorithm Comparison on {dataset_name}", fontsize=14, fontweight="bold")
    for bar, acc, n in zip(bars, accuracies, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2%}\n(n={n})",
            ha="center", va="bottom", fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

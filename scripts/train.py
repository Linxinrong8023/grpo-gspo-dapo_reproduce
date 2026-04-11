"""
scripts/train.py - RL 训练入口。
1111
用法:
    python scripts/train.py --config configs/grpo_math.json
    python scripts/train.py --config configs/grpo_math.json --max-steps 100
"""

import logging
import os
import sys
import traceback
import warnings

# 必须在 torch import 之前设置，解决显存碎片导致的 OOM
# 参考：https://pytorch.org/docs/stable/notes/cuda.html#memory-management
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# torch==2.4.0 的 torch.utils.checkpoint 仍会在内部调用已弃用的
# torch.cpu.amp.autocast(...)，开启 gradient checkpointing 时会反复刷屏。
# 这是上游实现细节，不影响训练结果；先精准屏蔽这一条，等后续统一升级 torch。
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated\.",
    category=FutureWarning,
    module=r"torch\.utils\.checkpoint",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_training_contrast.config import load_train_config
from src.post_training_contrast.trainer import run_training

logger = logging.getLogger(__name__)


def main() -> None:
    config = load_train_config()  # 此时 setup_run_logging() 已被调用，log 文件已打开
    logger.info("=" * 60)
    logger.info(
        "Training started  algo=%s  model=%s", config.algorithm_name, config.model
    )
    logger.info(
        "Config: output_dir=%s  dataset=%s", config.output_dir, config.dataset_path
    )
    logger.info("=" * 60)
    try:
        run_training(config)
        logger.info("Training finished successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        sys.exit(1)
    except Exception:
        # logger 可能已被 vLLM 破坏，同时用 print 直接输出到终端
        err_msg = traceback.format_exc()
        print("\n" + "=" * 60, flush=True)
        print("TRAINING CRASHED! Full traceback:", flush=True)
        print("=" * 60, flush=True)
        print(err_msg, flush=True)
        print("=" * 60, flush=True)
        logger.exception("Training crashed with unhandled exception:\n%s", err_msg)
        sys.exit(2)


if __name__ == "__main__":
    main()

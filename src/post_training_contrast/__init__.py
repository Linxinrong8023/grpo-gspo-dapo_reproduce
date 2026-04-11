import os
import sys
from types import ModuleType

# ── 默认启用 Hugging Face 离线模式，避免离线训练时出现无意义的远程探测 ──
# 仓库当前配置默认都指向本地模型目录；如果确实需要联网拉取，可在启动前显式导出 0 覆盖。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ── 自动修复：拦截 vLLM 依赖中 outlines < 0.1 版本导致的 pyairports 缺失报错 ──
# 这个问题在很多受限网络服务器（如 A800 实验室）中是顽疾。
# 该补丁通过在内存中动态构造 mock 模块，彻底免除安装 pyairports 的烦恼。

try:
    import pyairports
except ImportError:
    # 构造一个虚构的 pyairports 模块并注入到 sys.modules 中
    mock_pkg = ModuleType("pyairports")
    mock_sub = ModuleType("pyairports.airports")
    mock_sub.AIRPORT_LIST = []  # outlines 只用了这一个常量
    
    mock_pkg.airports = mock_sub
    sys.modules["pyairports"] = mock_pkg
    sys.modules["pyairports.airports"] = mock_sub

# ──────────────────────────────────────────────────────────────────────────────

"""
math_utils.py - 数学答案解析与符号等价比较工具。

这是一个**纯工具模块**，没有任何项目内部依赖。
被 reward.py 和 evaluator.py 共同导入。

包含：
  - parse_final_answer        从模型输出里提取最终答案
  - normalize_answer          轻量归一化（去空格、$、,）
  - symbolic_equals           有界符号/采样式答案等价比较
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

try:
    from func_timeout import FunctionTimedOut, func_timeout
except ImportError:
    class FunctionTimedOut(TimeoutError):
        pass

    def func_timeout(timeout, func, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))

try:
    from sympy import (
        Eq,
        Ge,
        Gt,
        I,
        Le,
        Lt,
        N,
        Pow,
        count_ops,
        preorder_traversal,
        sympify,
    )
    from latex2sympy2 import latex2sympy

    HAS_SYMBOLIC_BACKEND = True
except ImportError:
    HAS_SYMBOLIC_BACKEND = False


MAX_SYMBOLIC_INPUT_CHARS = 160
MAX_SYMBOLIC_TEXT_OPS = 32
MAX_SYMBOLIC_EXPR_OPS = 80
MAX_SYMBOLIC_VARIABLES = 4
MIN_VALID_SAMPLE_POINTS = 3
DEFAULT_REWARD_EVAL_TIMEOUT_SECONDS = 1.5
ALLOWED_MULTI_LETTER_SYMBOLS = {"e", "inf", "infty", "oo", "pi"}
SAMPLE_VALUE_ROWS = (
    (Fraction(-3, 1), Fraction(-3, 2), Fraction(-1, 1), Fraction(1, 2)),
    (Fraction(-1, 2), Fraction(2, 1), Fraction(3, 2), Fraction(5, 1)),
    (Fraction(2, 1), Fraction(5, 2), Fraction(5, 1), Fraction(7, 2)),
    (Fraction(5, 1), Fraction(-5, 2), Fraction(7, 1), Fraction(-3, 2)),
    (Fraction(7, 2), Fraction(11, 1), Fraction(-5, 2), Fraction(3, 1)),
)


# ── 答案提取 ─────────────────────────────────────────────────────


def parse_boxed_answer(text: str) -> str:
    """提取文本里最后一个 \\boxed{...} 的内容，支持嵌套花括号。"""
    start = text.rfind("\\boxed{")
    if start == -1:
        return ""

    cursor = start + len("\\boxed{")
    depth = 0
    chars: list[str] = []

    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            if depth == 0:
                return "".join(chars).strip()
            depth -= 1
            chars.append(char)
        else:
            chars.append(char)
        cursor += 1

    return ""


def parse_gsm8k_answer(text: str) -> str:
    """提取 GSM8K 格式里 #### 后面的最终答案。"""
    matched = re.search(r"####\s*(.+)", text)
    if not matched:
        return ""
    return matched.group(1).strip()


def parse_math_answer(text: str) -> str:
    """严格提取 \\boxed{...}；缺失正式答案标记时返回空。"""
    return parse_boxed_answer(text)


def _strip_think_block(text: str) -> str:
    """如果模型输出了 <think>...</think> 推理块，将其剥离，只保留推理块之后的部分。

    这样 parse_final_answer 只会在模型的"答案段"里找答案，
    不会被推理过程中出现的 \\boxed{} 或 #### 误导。
    """
    # 找最后一个 </think> 标签，取其之后的部分作为答案段
    think_end = text.rfind("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):].strip()
    # 如果只有 <think> 开头但没有闭合标签（截断情况），直接返回空
    # 以避免把推理过程误当成最终答案
    if text.strip().startswith("<think>"):
        return ""
    return text


def parse_final_answer(text: str, dataset_name: str) -> str:
    """按数据集类型选择提取策略的统一入口。

    支持两种输出格式：
    - 直接输出格式：直接在文本里找 boxed 或 ####
    - <think> 格式：先剥离 <think>...</think> 推理块，再在剩余部分里提取答案
    """
    # 首先剥离 <think> 推理块
    answer_segment = _strip_think_block(text)
    effective_text = answer_segment

    if dataset_name.lower() == "gsm8k":
        return parse_gsm8k_answer(effective_text)
    return parse_math_answer(effective_text)


# ── 归一化 ───────────────────────────────────────────────────────


def normalize_answer(answer: str) -> str:
    """对答案做最小必要的归一化。

    只去掉空白、美元符号、千分位逗号和常见 LaTeX 外壳。
    不做复杂的符号等价化（那由 symbolic_equals 负责）。
    """
    normalized = answer.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    return normalized


# ── 符号等价比较 ─────────────────────────────────────────────────


def _clean_for_symbolic(text: str) -> str:
    """去掉空白、美元符号和常见 LaTeX 外壳，让符号引擎更好解析。"""
    cleaned = _coerce_answer_like(text).strip()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\left", "")
    cleaned = cleaned.replace("\\right", "")
    cleaned = cleaned.replace("\\%", "%")
    cleaned = cleaned.replace("\\{", "{")
    cleaned = cleaned.replace("\\}", "}")

    # 移除常见的 LaTeX 字体和文本块修饰符，但保留内部的内容
    # 防止 \textbf{Evelyn} 或者 \mathrm{m} 之类的情况被误删或格式不匹配
    cleaned = re.sub(
        r"\\(?:text|textbf|textit|mathrm|mathit|mbox)\{([^{}]*)\}", r"\1", cleaned
    )

    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"\\frac([A-Za-z0-9])\{([^{}]+)\}", r"\\frac{\1}{\2}", cleaned)
    cleaned = re.sub(r"\\frac\{([^{}]+)\}([A-Za-z0-9])", r"\\frac{\1}{\2}", cleaned)
    cleaned = re.sub(r"\\frac([A-Za-z0-9])([A-Za-z0-9])", r"\\frac{\1}{\2}", cleaned)
    cleaned = cleaned.rstrip(".")
    return cleaned


def _split_tuple_items(text: str) -> list[str]:
    """识别形如 (a,b) 的坐标或元组答案，按最外层逗号切分。"""
    if len(text) < 3 or text[0] not in "([{" or text[-1] not in ")]}":
        return []

    items: list[str] = []
    current: list[str] = []
    depth = 0
    saw_comma = False

    for char in text[1:-1]:
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
            saw_comma = True
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        current.append(char)

    items.append("".join(current))
    if not saw_comma:
        return []
    return [item for item in items if item]


def _parse_numeric(text: str):
    """把整数、小数、百分数、a/b、\\frac{a}{b} 解析成可比较的有理数。"""
    cleaned = _clean_for_symbolic(text)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    positive_infinity = {
        "∞",
        "+∞",
        r"\infty",
        r"+\infty",
        "infty",
        "+infty",
        "infinity",
        "+infinity",
        "inf",
        "+inf",
        "oo",
        "+oo",
    }
    negative_infinity = {"-∞", r"-\infty", "-infty", "-infinity", "-inf", "-oo"}
    if lowered in positive_infinity:
        return Decimal("Infinity")
    if lowered in negative_infinity:
        return Decimal("-Infinity")
    if lowered in {"nan", "+nan", "-nan"}:
        return None

    if cleaned.endswith("%"):
        val = _parse_numeric(cleaned[:-1])
        return val / 100 if val is not None else None

    latex_frac = re.fullmatch(r"([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}", cleaned)
    if latex_frac:
        sign = -1 if latex_frac.group(1) == "-" else 1
        num = _parse_numeric(latex_frac.group(2))
        den = _parse_numeric(latex_frac.group(3))
        if num is None or den in (None, 0):
            return None
        return sign * num / den

    slash_frac = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", cleaned)
    if slash_frac:
        num, den = int(slash_frac.group(1)), int(slash_frac.group(2))
        return None if den == 0 else Fraction(num, den)

    try:
        decimal_value = Decimal(cleaned)
        if decimal_value.is_nan():
            return None
        if decimal_value.is_infinite():
            return decimal_value
        if abs(decimal_value.adjusted()) > 1000:
            return None
        return Fraction(decimal_value)
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None


def _numeric_equals(left: str, right: str) -> bool:
    """比较两个数值答案是否相等（支持分数/小数等价）。"""
    lv = _parse_numeric(left)
    rv = _parse_numeric(right)
    return lv is not None and rv is not None and lv == rv


def _contains_symbolic_variable(text: str) -> bool:
    """判断表达式里是否含有普通变量名，忽略 LaTeX 命令本身。"""
    without_latex_commands = re.sub(r"\\[A-Za-z]+", "", text)
    return re.search(r"[A-Za-z]", without_latex_commands) is not None


def _symbolic_text_too_complex(text: str) -> bool:
    """在调用 SymPy 前做轻量复杂度门控，避免坏样本拖死 reward。"""
    if len(text) > MAX_SYMBOLIC_INPUT_CHARS:
        return True
    op_count = sum(text.count(op) for op in ("+", "-", "*", "/", "^", "=", "<", ">"))
    if op_count > MAX_SYMBOLIC_TEXT_OPS:
        return True
    grouping_count = sum(text.count(ch) for ch in ("(", ")", "{", "}", "[", "]"))
    return grouping_count > MAX_SYMBOLIC_TEXT_OPS * 2


def _looks_like_symbolic_math(text: str) -> bool:
    """过滤自然语言答案，避免明显非数学文本进入 SymPy 慢路径。"""
    if re.fullmatch(r"[0-9A-Za-z\\{}()[\].,+\-*/^_=<>%|:∞]+", text) is None:
        return False
    without_latex_commands = re.sub(r"\\[A-Za-z]+", "", text)
    alpha_tokens = re.findall(r"[A-Za-z]+", without_latex_commands)
    return all(
        len(token) == 1 or token.lower() in ALLOWED_MULTI_LETTER_SYMBOLS
        for token in alpha_tokens
    )


def _should_try_sympy(left: str, right: str) -> bool:
    """判断是否值得进入有界符号路径。"""
    if _symbolic_text_too_complex(left) or _symbolic_text_too_complex(right):
        return False
    if not _looks_like_symbolic_math(left) or not _looks_like_symbolic_math(right):
        return False

    left_num = _parse_numeric(left)
    right_num = _parse_numeric(right)
    if left_num is not None and right_num is None:
        return not _contains_symbolic_variable(right)
    if right_num is not None and left_num is None:
        return not _contains_symbolic_variable(left)
    return True


def _tuple_equals(left: str, right: str) -> bool:
    """比较坐标或元组答案是否逐项相等。"""
    li = _split_tuple_items(_clean_for_symbolic(left))
    ri = _split_tuple_items(_clean_for_symbolic(right))
    if not li or not ri or len(li) != len(ri):
        return False
    return all(symbolic_equals(a, b) for a, b in zip(li, ri))


def _parse_symbolic(text: str):
    """尝试把答案解析成 SymPy 表达式（需要 sympy + latex2sympy2）。

    返回值可以是普通表达式（Number/Symbol/...）或 Relational（方程/不等式）。
    上层 _sympy_equals 会根据类型分路处理。
    """
    if not HAS_SYMBOLIC_BACKEND:
        return None
    from sympy.core.relational import Relational  # noqa: F401（供 isinstance 检查）

    cleaned = _clean_for_symbolic(text)
    if not cleaned:
        return None
    try:
        if "\\" in cleaned:
            result = latex2sympy(cleaned)
        else:
            relation = _parse_plaintext_relation(cleaned)
            if relation is not None:
                return relation
            result = sympify(_prepare_plaintext_symbolic(cleaned))
        return result
    except Exception:
        return None


def _sympy_expr_too_complex(expr) -> bool:
    try:
        return int(count_ops(expr)) > MAX_SYMBOLIC_EXPR_OPS
    except Exception:
        return True


def _sympy_symbol_count_too_high(*exprs) -> bool:
    try:
        symbols = set()
        for expr in exprs:
            symbols.update(getattr(expr, "free_symbols", set()))
        return len(symbols) > MAX_SYMBOLIC_VARIABLES
    except Exception:
        return True


def _expr_has_disallowed_constructs(expr) -> bool:
    """保守拒绝容易带来定义域/分支问题的表达式。"""
    try:
        if expr.has(I):
            return True
        if getattr(expr, "is_real", None) is False:
            return True
        for node in preorder_traversal(expr):
            if getattr(node, "is_Function", False):
                return True
        for power in expr.atoms(Pow):
            exponent = getattr(power, "exp", None)
            if getattr(exponent, "is_Rational", False) and exponent.q != 1:
                return True
        return False
    except Exception:
        return True


def _sympy_equals(left: str, right: str) -> bool:
    """用有界 SymPy 解析 + 采样做保守等价比较。

    分两条路处理：
    1. 两边都是普通表达式 → 采样/轻量 expand 判零
    2. 两边都是 Relational（方程/不等式）→ 比较 (lhs-rhs) 是否等价
    3. 一边是 Relational、另一边不是 → 类型不匹配，直接 False
    """
    if not HAS_SYMBOLIC_BACKEND:
        return False
    from sympy.core.relational import Equality, Relational

    try:
        le = _parse_symbolic(left)
        re_ = _parse_symbolic(right)
        if le is None or re_ is None:
            return False
        if _sympy_expr_too_complex(le) or _sympy_expr_too_complex(re_):
            return False
        if _sympy_symbol_count_too_high(le, re_):
            return False

        le_is_rel = isinstance(le, Relational)
        re_is_rel = isinstance(re_, Relational)
        le_bad = (
            _expr_has_disallowed_constructs(le.lhs)
            or _expr_has_disallowed_constructs(le.rhs)
            if le_is_rel
            else _expr_has_disallowed_constructs(le)
        )
        re_bad = (
            _expr_has_disallowed_constructs(re_.lhs)
            or _expr_has_disallowed_constructs(re_.rhs)
            if re_is_rel
            else _expr_has_disallowed_constructs(re_)
        )
        if le_bad or re_bad:
            return False

        if le_is_rel and re_is_rel:
            if isinstance(le, Equality) and isinstance(re_, Equality):
                return _equalities_match(le, re_)
            return _other_relationals_match(le, re_)

        # 情形 2：类型不匹配（一边是方程，另一边是值）
        if le_is_rel or re_is_rel:
            return False

        # 情形 3：两边都是普通表达式
        return _sympy_exprs_equal(le, re_)
    except Exception:
        return False


def symbolic_equals(predicted: str, ground_truth: str) -> bool:
    """统一的数学答案等价比较入口。

    比较顺序（由快到慢）：
      1. 字符串完全相同
      2. 数值等价（有理数精确比较）
      3. 元组/坐标逐项比较
      4. 有界符号采样等价（需要安装 sympy + latex2sympy2）
    """
    p = _clean_for_symbolic(predicted)
    g = _clean_for_symbolic(ground_truth)

    if not p or not g:
        return False
    if p == g:
        return True
    if _numeric_equals(p, g):
        return True
    if _tuple_equals(p, g):
        return True
    return _should_try_sympy(p, g) and _sympy_equals(p, g)


def _coerce_answer_like(value) -> str:
    """把答案包装类型转成普通 Python 字符串，避免 tensor-like 对象泄漏进判分。"""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _coerce_answer_like(value[0])

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item_value = item_method()
        except Exception:
            item_value = value
        if item_value is not value:
            return _coerce_answer_like(item_value)

    if value is None:
        return ""
    return str(value)


def _prepare_plaintext_symbolic(text: str) -> str:
    prepared = text.replace("^", "**")
    prepared = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", prepared)
    prepared = re.sub(r"(?<=\))(?=[A-Za-z0-9])", "*", prepared)
    prepared = re.sub(r"(?<=[A-Za-z0-9])(?=\()", "*", prepared)
    return prepared


def _parse_plaintext_relation(text: str):
    relation_match = re.search(r"(<=|>=|=|<|>)", text)
    if relation_match is None:
        return None
    if re.search(r"(<=|>=|=|<|>).*(<=|>=|=|<|>)", text[relation_match.end() :]):
        return None

    lhs_text = text[: relation_match.start()]
    rhs_text = text[relation_match.end() :]
    if not lhs_text or not rhs_text:
        return None

    lhs = sympify(_prepare_plaintext_symbolic(lhs_text))
    rhs = sympify(_prepare_plaintext_symbolic(rhs_text))
    relation_builders = {
        "=": Eq,
        "<": Lt,
        ">": Gt,
        "<=": Le,
        ">=": Ge,
    }
    return relation_builders[relation_match.group(1)](lhs, rhs)


def _expr_is_direct_zero(expr) -> bool:
    if expr == 0 or getattr(expr, "is_zero", None) is True:
        return True
    return False


def _expr_expand_is_zero(expr) -> bool:
    if _sympy_expr_too_complex(expr) or _expr_has_disallowed_constructs(expr):
        return False
    try:
        expanded = expr.expand()
    except Exception:
        return False
    return expanded == 0 or getattr(expanded, "is_zero", None) is True


def _value_is_finite_real(value) -> bool:
    try:
        if getattr(value, "free_symbols", set()):
            return False
        numeric = N(value)
    except Exception:
        return False
    if getattr(numeric, "is_finite", None) is not True:
        return False
    if getattr(numeric, "is_real", None) is False:
        return False
    try:
        if numeric.has(I):
            return False
    except Exception:
        return False
    return True


def _value_is_zero(value) -> bool:
    if value == 0 or getattr(value, "is_zero", None) is True:
        return True
    if not _value_is_finite_real(value):
        return False
    try:
        return bool(abs(N(value, 30)) <= 1e-9)
    except Exception:
        return False


def _sample_substitutions(symbols) -> list[dict]:
    ordered_symbols = sorted(symbols, key=lambda symbol: str(symbol))
    if len(ordered_symbols) > MAX_SYMBOLIC_VARIABLES:
        return []
    substitutions = []
    for row in SAMPLE_VALUE_ROWS:
        substitutions.append(
            {
                symbol: row[idx % len(row)]
                for idx, symbol in enumerate(ordered_symbols)
            }
        )
    return substitutions


def _evaluate_at(expr, substitutions):
    try:
        value = expr.subs(substitutions)
    except Exception:
        return None
    if not _value_is_finite_real(value):
        return None
    return value


def _sample_difference_is_zero(diff) -> bool:
    symbols = getattr(diff, "free_symbols", set())
    if not symbols:
        return _value_is_zero(diff)
    if _sympy_symbol_count_too_high(diff) or _expr_has_disallowed_constructs(diff):
        return False

    valid_points = 0
    for substitutions in _sample_substitutions(symbols):
        value = _evaluate_at(diff, substitutions)
        if value is None:
            continue
        valid_points += 1
        if not _value_is_zero(value):
            return False
    return valid_points >= MIN_VALID_SAMPLE_POINTS


def _sympy_exprs_equal(left_expr, right_expr) -> bool:
    diff = left_expr - right_expr
    if _expr_is_direct_zero(diff):
        return True
    if _sympy_expr_too_complex(diff) or _sympy_symbol_count_too_high(diff):
        return False
    if _expr_has_disallowed_constructs(diff):
        return False
    if getattr(diff, "is_number", None) is True:
        return _value_is_zero(diff)
    if _sample_difference_is_zero(diff):
        return True
    return _expr_expand_is_zero(diff)


def _constant_nonzero_ratio(left_expr, right_expr) -> bool:
    if _expr_is_direct_zero(left_expr) and _expr_is_direct_zero(right_expr):
        return True
    if _expr_is_direct_zero(left_expr) or _expr_is_direct_zero(right_expr):
        return False
    if _sympy_expr_too_complex(left_expr) or _sympy_expr_too_complex(right_expr):
        return False
    if _sympy_symbol_count_too_high(left_expr, right_expr):
        return False
    if _expr_has_disallowed_constructs(left_expr) or _expr_has_disallowed_constructs(
        right_expr
    ):
        return False

    symbols = set(getattr(left_expr, "free_symbols", set()))
    symbols.update(getattr(right_expr, "free_symbols", set()))
    first_ratio = None
    valid_points = 0
    for substitutions in _sample_substitutions(symbols):
        left_value = _evaluate_at(left_expr, substitutions)
        right_value = _evaluate_at(right_expr, substitutions)
        if left_value is None or right_value is None:
            continue
        left_zero = _value_is_zero(left_value)
        right_zero = _value_is_zero(right_value)
        if left_zero and right_zero:
            continue
        if left_zero or right_zero:
            return False
        try:
            ratio = left_value / right_value
        except Exception:
            continue
        if not _value_is_finite_real(ratio) or _value_is_zero(ratio):
            continue
        if first_ratio is None:
            first_ratio = ratio
        elif not _value_is_zero(ratio - first_ratio):
            return False
        valid_points += 1
    return first_ratio is not None and valid_points >= MIN_VALID_SAMPLE_POINTS


def _equalities_match(left_rel, right_rel) -> bool:
    left_zero = left_rel.lhs - left_rel.rhs
    right_zero = right_rel.lhs - right_rel.rhs

    if _sympy_exprs_equal(left_zero, right_zero):
        return True
    if _sympy_exprs_equal(left_zero, -right_zero):
        return True
    return _constant_nonzero_ratio(left_zero, right_zero)


def _other_relationals_match(left_rel, right_rel) -> bool:
    mirror_types = {
        Lt: Gt,
        Gt: Lt,
        Le: Ge,
        Ge: Le,
    }
    if type(left_rel) is type(right_rel):
        return _sympy_exprs_equal(left_rel.lhs, right_rel.lhs) and _sympy_exprs_equal(
            left_rel.rhs, right_rel.rhs
        )
    mirror_type = mirror_types.get(type(left_rel))
    if mirror_type is not type(right_rel):
        return False
    return _sympy_exprs_equal(left_rel.lhs, right_rel.rhs) and _sympy_exprs_equal(
        left_rel.rhs, right_rel.lhs
    )


def batch_evaluate_rewards(
    predicted_list,
    ground_truth_list,
) -> list[float]:
    """单线程批量比较答案。"""
    predicted_items = list(predicted_list)
    ground_truth_items = list(ground_truth_list)
    if len(predicted_items) != len(ground_truth_items):
        raise ValueError("predicted_list 和 ground_truth_list 长度必须一致")

    results: list[float] = []
    for predicted, ground_truth in zip(predicted_items, ground_truth_items):
        predicted_text = _coerce_answer_like(predicted)
        ground_truth_text = _coerce_answer_like(ground_truth)
        try:
            is_correct = func_timeout(
                DEFAULT_REWARD_EVAL_TIMEOUT_SECONDS,
                symbolic_equals,
                args=(predicted_text, ground_truth_text),
            )
            score = 1.0 if is_correct else 0.0
        except FunctionTimedOut:
            score = 0.0
        except Exception:
            score = 0.0
        results.append(score)
    return results

"""评估引擎 - 统一模型问答评估、打分与对比报告"""
import math
import re
from typing import Callable, Dict, List, Optional


_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def _char_tokens(text: str) -> List[str]:
    """按字符切分（对中文效果更好）：去掉空白和标点，保留字符"""
    return [c for c in text.strip().lower() if c.isalnum()]


def _word_tokens(text: str) -> List[str]:
    """按词切分（对英文/代码效果好）"""
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_text_similarity(text1: str, text2: str) -> float:
    """字符级 + 词级加权相似度（F1），0~1"""
    if not text1 or not text2:
        return 0.0
    c1, c2 = set(_char_tokens(text1)), set(_char_tokens(text2))
    w1, w2 = set(_word_tokens(text1)), set(_word_tokens(text2))

    def f1(a, b):
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter == 0:
            return 0.0
        return 2 * inter / (len(a) + len(b))

    char_f1 = f1(c1, c2)
    word_f1 = f1(w1, w2)
    # 纯中文场景词级切分为空，退化为纯字符相似度；
    # 中英混合/代码场景则字符级与词级加权
    if not w1 and not w2:
        return round(char_f1, 4)
    if not c1 and not c2:
        return round(word_f1, 4)
    return round(0.6 * char_f1 + 0.4 * word_f1, 4)


def _ngrams(text: str, n: int) -> List[tuple]:
    """按词（中文退化为字符）切分 n-gram"""
    toks = _word_tokens(text) or _char_tokens(text)
    if len(toks) < n:
        return [tuple(toks)]
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def bleu_1(reference: str, candidate: str) -> float:
    """BLEU-1（含长度惩罚），0~1"""
    if not reference or not candidate:
        return 0.0
    ref_ngrams = set(_ngrams(reference, 1))
    cand_ngrams = _ngrams(candidate, 1)
    if not cand_ngrams:
        return 0.0
    precision = sum(1 for g in cand_ngrams if g in ref_ngrams) / len(cand_ngrams)
    brevity = math.exp(1 - len(ref_ngrams) / len(cand_ngrams)) \
        if len(cand_ngrams) < len(ref_ngrams) else 1.0
    return round(precision * brevity, 4)


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F1（最长公共子序列），0~1"""
    ref_toks = _word_tokens(reference) or _char_tokens(reference)
    cand_toks = _word_tokens(candidate) or _char_tokens(candidate)
    if not ref_toks or not cand_toks:
        return 0.0
    m, n = len(ref_toks), len(cand_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_toks[i - 1] == cand_toks[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / n, lcs / m
    return round(2 * precision * recall / (precision + recall), 4)


def score_answer(answer: str,
                 reference: Optional[str] = None,
                 baseline_answer: Optional[str] = None) -> Dict[str, float]:
    """对单条回答给出多维度分数（0~100）

    维度：
    - 回答完整性: 长度是否充分（相对参考答案长度，无参考时按 80 字符目标）
    - 内容覆盖率: 与参考答案的相似度；无参考时返回 0（表示“未评估”）
    - 语言流畅度: 词型-词次比（越高说明信息密度/用词多样性越好）
    - 基线一致性: 与蒸馏/剪枝前回答的相似度（衡量输出是否退化）
    """
    answer = (answer or "").strip()
    target_len = max(60, len((reference or "").strip())) if reference else 80
    completeness = min(100.0, len(answer) / target_len * 100)

    if reference:
        # 字符/词级 F1 + BLEU-1 + ROUGE-L 三合一，避免单一指标偏差
        coverage = round((
            compute_text_similarity(answer, reference)
            + bleu_1(reference, answer)
            + rouge_l(reference, answer)
        ) / 3 * 100, 1)
    else:
        coverage = 0.0

    words = _word_tokens(answer)
    chars = _char_tokens(answer)
    if words:
        fluency = min(100.0, len(set(words)) / len(words) * 100)
    elif chars:
        fluency = min(100.0, len(set(chars)) / len(chars) * 100)
    else:
        fluency = 0.0

    stability = compute_text_similarity(answer, baseline_answer) * 100 if baseline_answer else 0.0

    return {
        "回答完整性": round(completeness, 1),
        "内容覆盖率": round(coverage, 1),
        "语言流畅度": round(fluency, 1),
        "基线一致性": round(stability, 1),
    }


def compute_scores(baseline: List[Dict],
                   after: Optional[List[Dict]] = None,
                   references: Optional[List[str]] = None) -> Dict:
    """对基线/处理后两组回答计算多维度平均分

    返回结构（可直接用于雷达图）：
    {
        "dimensions": [...],
        "baseline": [...],
        "after": [...],
        "per_question": [{question, baseline_score, after_score, reference}...],
    }
    """
    baseline = baseline or []
    after = after or []
    references = references or []

    per_question = []
    for i, b in enumerate(baseline):
        ref = references[i] if i < len(references) else None
        a = after[i] if i < len(after) else {}
        b_score = score_answer(b.get("answer", ""), reference=ref)
        a_score = score_answer(a.get("answer", ""), reference=ref,
                               baseline_answer=b.get("answer", ""))
        per_question.append({
            "question": b.get("question", ""),
            "image": b.get("image", ""),
            "baseline": b_score,
            "after": a_score,
            "reference": ref or "",
        })

    dims = ["回答完整性", "内容覆盖率", "语言流畅度", "基线一致性"]

    def avg(key, items):
        vals = [s[key] for s in items if s]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    return {
        "dimensions": dims,
        "baseline": [avg(d, [p["baseline"] for p in per_question]) for d in dims],
        "after": [avg(d, [p["after"] for p in per_question]) for d in dims],
        "per_question": per_question,
    }


def compute_clip_alignment(items: List[Dict]) -> Optional[List[float]]:
    """可选：CLIP 图像-文本对齐评分（0~100，衡量描述是否与图片语义一致）

    items: [{"image_path": ..., "answer": ...}, ...]
    依赖 transformers + 首次使用时下载 openai/clip-vit-base-patch32；
    无网络/无依赖/任一图片不可用返回 None（调用方应优雅降级）。
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if not items or any(not it.get("image_path") for it in items):
        return None
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        from PIL import Image
    except ImportError:
        return None

    if _CLIP_MODEL is None:
        try:
            _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception:
            return None
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _CLIP_MODEL.to(device)
    _CLIP_MODEL.eval()

    scores = []
    try:
        def _feat_vec(feat):
            """兼容新旧 transformers：BaseModelOutput / 张量 / tuple"""
            if hasattr(feat, "pooler_output"):
                return feat.pooler_output
            if hasattr(feat, "last_hidden_state"):
                return feat.last_hidden_state[:, 0]
            if isinstance(feat, tuple) and len(feat) > 0:
                return feat[0]
            return feat

        with torch.no_grad():
            for it in items:
                image = Image.open(it["image_path"]).convert("RGB")
                text = (it.get("answer") or "").strip()[:77] or "图片"
                inputs = _CLIP_PROCESSOR(text=[text], images=[image],
                                         return_tensors="pt", padding=True,
                                         truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                image_feat = _feat_vec(_CLIP_MODEL.get_image_features(
                    pixel_values=inputs["pixel_values"]
                ))
                text_feat = _feat_vec(_CLIP_MODEL.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ))
                image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                sim = (image_feat @ text_feat.T).item()
                scores.append(round(max(0.0, min(100.0, (sim + 1) / 2 * 100)), 1))
    except Exception:
        return None
    return scores


def simple_eval(model, tokenizer, questions: List[str],
                device: Optional[str] = None,
                max_new_tokens: int = 64,
                cancel_check: Optional[Callable[[], None]] = None) -> List[Dict]:
    """对模型进行简单问答评估（引擎统一入口）"""
    import torch  # 延迟导入：评分/报告等纯函数不依赖 torch
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"
    model.eval()
    results = []
    for q in questions:
        if cancel_check:
            cancel_check()
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        results.append({"question": q, "answer": response})
    return results


def generate_comparison_report(baseline: List[Dict],
                               after: Optional[List[Dict]] = None,
                               scores: Optional[Dict] = None,
                               teacher: Optional[List[Dict]] = None,
                               model_name_a: str = "基线",
                               model_name_b: str = "处理后",
                               extra_info: Optional[Dict] = None) -> str:
    """生成评估对比报告（Markdown）"""
    baseline = baseline or []
    after = after or []
    teacher = teacher or []
    scores = scores or {}
    extra_info = extra_info or {}
    has_images = any(b.get("image") for b in baseline if b)

    lines = [
        "# 模型能力对比报告",
        "",
        f"- 实验类型: {extra_info.get('type', '-')}",
        f"- 生成时间: {extra_info.get('time', '-')}",
        f"- 蒸馏数据量: {extra_info.get('num_data', '-')}",
        "",
        "## 维度评分",
        "",
        "| 维度 | {0} | {1} |".format(model_name_a, model_name_b),
        "|------|------|------|",
    ]
    dims = scores.get("dimensions", [])
    b_scores = scores.get("baseline", [])
    a_scores = scores.get("after", [])
    for i, d in enumerate(dims):
        bv = b_scores[i] if i < len(b_scores) else "-"
        av = a_scores[i] if i < len(a_scores) else "-"
        lines.append(f"| {d} | {bv} | {av} |")

    lines.extend([
        "",
        "## 逐问题回答对比",
        "",
    ])
    header = "| " + ("图片 | " if has_images else "") + "问题"
    if teacher:
        header += " | 教师"
    header += " | {0} | {1} |".format(model_name_a, model_name_b)
    sep = "|" + "------|" * (header.count("|") - 1)
    lines.append(header)
    lines.append(sep)
    for i, b in enumerate(baseline):
        a_ans = after[i].get("answer", "") if i < len(after) else ""
        row = "| "
        if has_images:
            row += f"{b.get('image', '-')} | "
        row += f"{b.get('question','')[:30]} | "
        if teacher:
            t_ans = teacher[i].get("answer", "") if i < len(teacher) else ""
            row += f"{t_ans[:100]} | "
        row += f"{b.get('answer','')[:100]} | {a_ans[:100]} |"
        lines.append(row)

    if has_images:
        # 按图汇总识别详情
        lines.extend(["", "## 逐图识别详情"])
        per_image = {}
        for i, b in enumerate(baseline):
            per_image.setdefault(b.get("image", "未知图片"), []).append(i)
        for img, idxs in per_image.items():
            lines.append(f"### {img}")
            for i in idxs:
                a_ans = after[i].get("answer", "") if i < len(after) else ""
                lines.append(f"- **问题**: {baseline[i].get('question','')}")
                if teacher and i < len(teacher):
                    lines.append(f"  - 教师: {teacher[i].get('answer','')[:200]}")
                lines.append(f"  - {model_name_a}: {baseline[i].get('answer','')[:200]}")
                lines.append(f"  - {model_name_b}: {a_ans[:200]}")

    if scores.get("per_question"):
        lines.extend(["", "## 逐问题评分", "", "| 问题 | 完整性(B/A) | 覆盖率(B/A) | 流畅度(B/A) | 一致性(B/A) |"])
        for p in scores["per_question"]:
            b, a = p["baseline"], p["after"]
            lines.append(
                f"| {p['question'][:24]} | {b['回答完整性']}/{a['回答完整性']} "
                f"| {b['内容覆盖率']}/{a['内容覆盖率']} "
                f"| {b['语言流畅度']}/{a['语言流畅度']} "
                f"| {b['基线一致性']}/{a['基线一致性']} |"
            )

    lines.extend(["", "## 分析总结"])
    if dims:
        for i, d in enumerate(dims):
            bv = b_scores[i] if i < len(b_scores) else 0
            av = a_scores[i] if i < len(a_scores) else 0
            delta = av - bv
            arrow = "↑" if delta > 1 else ("↓" if delta < -1 else "→")
            lines.append(f"- {d}: {bv} → {av} {arrow}")
    return "\n".join(lines)


class EvaluationEngine:
    """评估引擎（类接口，兼容引擎/页面按类调用的写法）"""

    simple_eval = staticmethod(simple_eval)
    compute_scores = staticmethod(compute_scores)
    generate_comparison_report = staticmethod(generate_comparison_report)
    compute_text_similarity = staticmethod(compute_text_similarity)
    bleu_1 = staticmethod(bleu_1)
    rouge_l = staticmethod(rouge_l)
    score_answer = staticmethod(score_answer)

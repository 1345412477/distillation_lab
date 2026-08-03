"""评估引擎评分函数单元测试（不依赖 torch / PyQt）"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.evaluation import (
    compute_text_similarity,
    bleu_1,
    rouge_l,
    score_answer,
    compute_scores,
    generate_comparison_report,
    compute_clip_alignment,
)


class TestTextSimilarity(unittest.TestCase):
    def test_identical(self):
        self.assertGreater(compute_text_similarity("深度学习是机器学习的分支", "深度学习是机器学习的分支"), 0.99)

    def test_disjoint(self):
        self.assertEqual(compute_text_similarity("苹果", "香蕉橘子葡萄"), 0.0)

    def test_partial_overlap(self):
        sim = compute_text_similarity("HTTP 和 HTTPS 的区别", "HTTP 与 HTTPS 有什么不同")
        self.assertGreater(sim, 0.1)

    def test_empty(self):
        self.assertEqual(compute_text_similarity("", "abc"), 0.0)


class TestReferenceMetrics(unittest.TestCase):
    def test_bleu_identical(self):
        text = "这是一段用于测试的参考回答文本内容"
        self.assertGreater(bleu_1(text, text), 0.99)

    def test_bleu_disjoint(self):
        self.assertEqual(bleu_1("苹果香蕉", "汽车飞机"), 0.0)

    def test_rouge_identical(self):
        text = "图片中有一只猫坐在窗台上晒太阳"
        self.assertGreater(rouge_l(text, text), 0.99)

    def test_rouge_partial(self):
        score = rouge_l("一只猫坐在窗台上", "窗台上有一只猫")
        self.assertGreater(score, 0.3)

    def test_rouge_empty(self):
        self.assertEqual(rouge_l("", "abc"), 0.0)


class TestScoreAnswer(unittest.TestCase):
    def test_full_reference_match(self):
        ref = ("这是一段用于测试回答完整度的参考回答文本，"
               "内容需要足够长以超过六十个字符的评分下限，"
               "这样才能验证完整度评分的正确性。")
        ans = ref
        s = score_answer(ans, reference=ref)
        self.assertGreater(s["内容覆盖率"], 90)
        self.assertGreater(s["回答完整性"], 90)

    def test_short_answer(self):
        s = score_answer("好")
        self.assertLess(s["回答完整性"], 30)

    def test_empty_answer(self):
        s = score_answer("")
        self.assertEqual(s["回答完整性"], 0.0)

    def test_stability_vs_baseline(self):
        s = score_answer("答案文本内容", baseline_answer="答案文本内容")
        self.assertGreater(s["基线一致性"], 90)


class TestComputeScores(unittest.TestCase):
    def test_structure(self):
        baseline = [
            {"question": "Q1", "answer": "这是一个比较长的回答内容，用来测试评分"},
            {"question": "Q2", "answer": "短答"},
        ]
        after = [
            {"question": "Q1", "answer": "这是一个比较长的回答内容，用来测试评分"},
            {"question": "Q2", "answer": "短答"},
        ]
        scores = compute_scores(baseline, after)
        self.assertEqual(len(scores["dimensions"]), 4)
        self.assertEqual(len(scores["baseline"]), 4)
        self.assertEqual(len(scores["after"]), 4)
        self.assertEqual(len(scores["per_question"]), 2)
        # 完全相同的回答：一致性应很高
        self.assertGreater(scores["after"][3], 90)


class TestReport(unittest.TestCase):
    def test_generate(self):
        baseline = [{"question": "Q1", "answer": "回答一"}]
        after = [{"question": "Q1", "answer": "回答二"}]
        report = generate_comparison_report(baseline, after, compute_scores(baseline, after))
        self.assertIn("模型能力对比报告", report)
        self.assertIn("回答完整性", report)

    def test_generate_with_teacher_and_images(self):
        baseline = [
            {"question": "图里有什么", "answer": "一只猫", "image": "cat.png"},
            {"question": "场景如何", "answer": "窗台", "image": "cat.png"},
        ]
        after = [
            {"question": "图里有什么", "answer": "一只猫在窗台", "image": "cat.png"},
            {"question": "场景如何", "answer": "阳光窗台", "image": "cat.png"},
        ]
        teacher = [
            {"question": "图里有什么", "answer": "一只橘猫", "image": "cat.png"},
            {"question": "场景如何", "answer": "明亮窗台", "image": "cat.png"},
        ]
        report = generate_comparison_report(
            baseline, after, compute_scores(baseline, after), teacher=teacher
        )
        self.assertIn("| 图片 |", report)
        self.assertIn("| 教师 |", report)
        self.assertIn("## 逐图识别详情", report)
        self.assertIn("### cat.png", report)
        self.assertIn("一只橘猫", report)

    def test_clip_alignment_fast_path(self):
        # 缺少 image_path 时应立即返回 None，不触发模型加载
        self.assertIsNone(compute_clip_alignment([{"answer": "一只猫"}]))
        self.assertIsNone(compute_clip_alignment([]))


if __name__ == "__main__":
    unittest.main()

"""蒸馏引擎辅助函数测试（生成参数/优化器/调度器；需要 torch）"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from app.engines.distillation import DistillationEngine
except ImportError:
    torch = None


class DummyWorker:
    def __init__(self):
        self.logs = []

    def emit_phase(self, *args):
        pass

    def emit_sample(self, *args):
        pass

    def check_cancelled(self):
        pass


@unittest.skipIf(torch is None, "需要 torch")
class TestDistillHelpers(unittest.TestCase):
    def test_generation_kwargs_sampling(self):
        cfg = {
            "temperature": 0.5, "top_p": 0.8, "top_k": 20,
            "repetition_penalty": 1.2, "num_beams": 1, "max_new_tokens": 100,
        }
        kw = DistillationEngine(cfg, DummyWorker())._generation_kwargs()
        self.assertEqual(kw["temperature"], 0.5)
        self.assertEqual(kw["top_k"], 20)
        self.assertEqual(kw["repetition_penalty"], 1.2)
        self.assertIs(kw["do_sample"], True)

    def test_generation_kwargs_beam(self):
        kw = DistillationEngine({"num_beams": 4}, DummyWorker())._generation_kwargs()
        self.assertEqual(kw["num_beams"], 4)
        self.assertIs(kw["do_sample"], False)

    def test_optimizer_selection(self):
        params = [torch.nn.Parameter(torch.zeros(2))]
        engine = DistillationEngine({"optimizer": "sgd"}, DummyWorker())
        self.assertIsInstance(engine._build_optimizer(params), torch.optim.SGD)
        engine = DistillationEngine({"optimizer": "adamw"}, DummyWorker())
        self.assertIsInstance(engine._build_optimizer(params), torch.optim.AdamW)

    def test_scheduler(self):
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))])
        engine = DistillationEngine({"warmup_steps": 10}, DummyWorker())
        self.assertIsNotNone(engine._build_scheduler(opt, 100))
        engine = DistillationEngine({"warmup_steps": 0}, DummyWorker())
        self.assertIsNone(engine._build_scheduler(opt, 100))

    def test_quant_config_params(self):
        engine = DistillationEngine({"use_4bit": False}, DummyWorker())
        q, kw = engine._get_quantization_config()
        self.assertIsNone(q)
        self.assertEqual(kw, {})


if __name__ == "__main__":
    unittest.main()

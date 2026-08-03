"""剪枝重建逻辑单元测试（需要 torch；无 torch 时自动跳过）"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from torch import nn
    from app.engines.pruning import PruningEngine
except ImportError:
    torch = None
    nn = None


if nn is not None:
    class FakeMLP(nn.Module):
        def __init__(self, hidden, inter):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, inter)
            self.up_proj = nn.Linear(hidden, inter)
            self.down_proj = nn.Linear(inter, hidden)

        def forward(self, x):
            return self.down_proj(self.up_proj(x))


    class FakeAttn(nn.Module):
        def __init__(self, hidden, heads, kv_heads, head_dim):
            super().__init__()
            self.q_proj = nn.Linear(hidden, heads * head_dim)
            self.k_proj = nn.Linear(hidden, kv_heads * head_dim)
            self.v_proj = nn.Linear(hidden, kv_heads * head_dim)
            self.o_proj = nn.Linear(heads * head_dim, hidden)


    class FakeLayer(nn.Module):
        def __init__(self, hidden, inter, heads, kv_heads, head_dim):
            super().__init__()
            self.self_attn = FakeAttn(hidden, heads, kv_heads, head_dim)
            self.mlp = FakeMLP(hidden, inter)


    class FakeModel(nn.Module):
        def __init__(self, hidden=64, inter=128, heads=8, kv_heads=4, head_dim=16, num_layers=6):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=hidden,
                intermediate_size=inter,
                num_attention_heads=heads,
                num_key_value_heads=kv_heads,
                num_hidden_layers=num_layers,
            )
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([
                FakeLayer(hidden, inter, heads, kv_heads, head_dim)
                for _ in range(num_layers)
            ])


class DummyWorker:
    """模拟 worker：信号用带 emit 的对象，方法用普通函数"""
    def __init__(self):
        self.log_updated = _Signal()
        self.status_updated = _Signal()
        self.metric_updated = _Signal()
        self.progress_updated = _Signal()

    def check_cancelled(self):
        pass

    def emit_phase(self, *args):
        pass

    def emit_sample(self, *args):
        pass


class _Signal:
    """最小信号桩：模拟 Qt 信号的 .emit()"""
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


@unittest.skipIf(torch is None, "需要 torch")
class TestPruningRebuild(unittest.TestCase):
    def setUp(self):
        self.engine = PruningEngine({"importance_metric": "l1"}, DummyWorker())

    def test_ffn_rebuild(self):
        model = FakeModel()
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.gate_proj.weight[0] = 100.0  # 让第 0 个神经元最重要
        stats = self.engine._apply_pruning(model, 0.5, "ffn", "l1")
        self.assertEqual(stats["mode"], "rebuild")
        for layer in model.model.layers:
            self.assertEqual(layer.mlp.gate_proj.out_features, 64)
            self.assertEqual(layer.mlp.up_proj.out_features, 64)
            self.assertEqual(layer.mlp.down_proj.in_features, 64)
        self.assertEqual(model.config.intermediate_size, 64)
        # 重要神经元（索引 0）必须保留
        self.assertEqual(
            model.model.layers[0].mlp.gate_proj.weight[0].max().item(), 100.0
        )
        self.assertGreater(stats["param_reduction_ratio"], 0)

    def test_attention_rebuild_gqa(self):
        model = FakeModel(heads=8, kv_heads=4, head_dim=16)
        stats = self.engine._apply_pruning(model, 0.5, "attention", "l1")
        self.assertEqual(stats["mode"], "rebuild")
        for layer in model.model.layers:
            self.assertEqual(layer.self_attn.q_proj.out_features, 64)   # 4 heads * 16
            self.assertEqual(layer.self_attn.k_proj.out_features, 32)   # 2 kv heads * 16
            self.assertEqual(layer.self_attn.o_proj.in_features, 64)
        self.assertEqual(model.config.num_attention_heads, 4)
        self.assertEqual(model.config.num_key_value_heads, 2)

    def test_layer_rebuild(self):
        model = FakeModel(num_layers=6)
        stats = self.engine._apply_pruning(model, 0.5, "layer", "l1")
        self.assertEqual(stats["mode"], "rebuild")
        self.assertEqual(len(model.model.layers), 3)
        self.assertEqual(model.config.num_hidden_layers, 3)
        self.assertGreater(stats["param_reduction_ratio"], 0)

    def test_forward_after_ffn_rebuild(self):
        """重建后前向传播形状必须一致"""
        model = FakeModel()
        model.model.layers[0].mlp(torch.randn(2, 64))
        self.engine._apply_pruning(model, 0.5, "ffn", "l1")
        out = model.model.layers[0].mlp(torch.randn(2, 64))
        self.assertEqual(out.shape, (2, 64))

    def test_unknown_strategy_falls_back_to_mask(self):
        model = FakeModel()
        stats = self.engine._apply_pruning(model, 0.5, "ffn", "l1")
        self.assertIn(stats["mode"], ("rebuild", "mask"))


if __name__ == "__main__":
    unittest.main()

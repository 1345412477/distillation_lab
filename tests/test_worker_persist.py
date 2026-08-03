"""worker 结果持久化逻辑测试（需要 PyQt6）"""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.workers.train_worker import ExperimentWorker
    from app.workers import train_worker as tw
except ImportError:
    ExperimentWorker = None
    tw = None


class FakeDB:
    def __init__(self):
        self.calls = []

    def update_experiment_metrics(self, *args):
        self.calls.append(("metrics", args))

    def update_experiment_results(self, *args, **kwargs):
        self.calls.append(("results", args, kwargs))

    def update_experiment_status(self, *args):
        self.calls.append(("status", args))


@unittest.skipIf(ExperimentWorker is None, "需要 PyQt6")
class TestWorkerPersist(unittest.TestCase):
    def test_accumulate_metric(self):
        worker = ExperimentWorker("distillation", {"exp_id": 7})
        worker._accumulate_metric("loss", 1.5, 0)
        worker._accumulate_metric("loss", 1.2, 1)
        worker._accumulate_metric("speed", 2.0, 1)  # 不应记录
        self.assertEqual(worker._metrics, [{"epoch": 0, "loss": 1.5}, {"epoch": 1, "loss": 1.2}])

    def test_persist_completed(self):
        worker = ExperimentWorker("pruning", {"exp_id": 9})
        worker.db = FakeDB()
        worker._persist_completed({"output_path": "/out", "report_path": "/rep"})
        kinds = [c[0] for c in worker.db.calls]
        self.assertEqual(kinds, ["metrics", "results", "status"])
        self.assertEqual(worker.db.calls[1][2]["output_path"], "/out")
        self.assertEqual(worker._last_status, "completed")

    def test_persist_no_exp_id(self):
        worker = ExperimentWorker("distillation", {})
        worker.db = FakeDB()
        worker._persist_completed({})
        self.assertEqual(worker.db.calls, [])

    def test_batch_summary_writes_markdown(self):
        class FakeRowDB:
            def get_experiment_by_id(self, exp_id):
                return {
                    "status": "completed",
                    "output_path": "/out/exp",
                    "eval_results": json.dumps({
                        "scores": {
                            "dimensions": ["回答完整性", "内容覆盖率"],
                            "baseline": [50.0, 40.0],
                            "after": [70.0, 60.0],
                        }
                    }),
                }

        td = tempfile.TemporaryDirectory()
        old_dir = tw.EXPERIMENT_DIR
        tw.EXPERIMENT_DIR = Path(td.name)
        try:
            worker = tw.BatchWorker([{"exp_id": 1, "_batch_name": "模板A"}])
            worker.db = FakeRowDB()
            worker.log_updated = _SignalStub()
            worker._write_batch_summary()
            files = list(Path(td.name).glob("batch_summary_*.md"))
            self.assertEqual(len(files), 1)
            content = files[0].read_text(encoding="utf-8")
            self.assertIn("模板A", content)
            self.assertIn("回答完整性: 50.0→70.0↑", content)
        finally:
            tw.EXPERIMENT_DIR = old_dir
            td.cleanup()


class _SignalStub:
    def emit(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()

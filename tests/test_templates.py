"""实验模板存储单元测试"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import template_store


class TestTemplateStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        template_store.TEMPLATE_DIR = Path(self._tmp.name)
        template_store.TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip(self):
        cfg = {"lora_r": 32, "epochs": 5, "top_k": 20, "image_prompts": ["p1", "p2"]}
        name = template_store.save_template("实验A", cfg)
        self.assertEqual(name, "实验A")
        self.assertEqual(template_store.load_template("实验A"), cfg)
        names = [t["name"] for t in template_store.list_templates()]
        self.assertIn("实验A", names)
        self.assertTrue(template_store.delete_template("实验A"))
        self.assertEqual(template_store.list_templates(), [])

    def test_safe_name(self):
        name = template_store.save_template("a/b:c", {"x": 1})
        self.assertNotIn("/", name)
        self.assertEqual(template_store.load_template(name)["x"], 1)


if __name__ == "__main__":
    unittest.main()

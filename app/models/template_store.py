"""实验模板存储 - 参数组合保存/加载/复用（便于批量调参）"""
import datetime
import json
import os
from ..utils.config import DATA_DIR


TEMPLATE_DIR = DATA_DIR / "templates"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    """模板文件名安全化"""
    return "".join(c for c in name.strip() if c not in '\\/:*?"<>|').strip() or "未命名模板"


def save_template(name: str, config: dict) -> str:
    """保存参数组合为模板，返回模板名"""
    name = _safe_name(name)
    template = {
        "name": name,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
    }
    path = TEMPLATE_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    return name


def list_templates() -> list:
    """列出全部模板（按保存时间倒序）"""
    templates = []
    if not TEMPLATE_DIR.is_dir():
        return templates
    for f in sorted(TEMPLATE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                tpl = json.load(fh)
            templates.append({
                "name": tpl.get("name", f.stem),
                "saved_at": tpl.get("saved_at", ""),
                "config": tpl.get("config", {}),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return templates


def load_template(name: str) -> dict:
    """按名称加载模板配置"""
    name = _safe_name(name)
    path = TEMPLATE_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"模板不存在: {name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("config", {})


def delete_template(name: str) -> bool:
    """删除模板，返回是否成功"""
    name = _safe_name(name)
    path = TEMPLATE_DIR / f"{name}.json"
    if path.is_file():
        path.unlink()
        return True
    return False

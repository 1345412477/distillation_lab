"""后台训练工作线程 - 通过信号/槽机制与 UI 通信 + 文件日志持久化"""
import traceback
from PyQt6.QtCore import QThread, pyqtSignal
from ..engines.distillation import DistillationEngine
from ..engines.pruning import PruningEngine
from ..models.db_manager import DBManager
from ..utils.exceptions import UserStopped
from ..utils.config import EXPERIMENT_DIR
from ..utils.file_logger import (
    setup_experiment_logger,
    write_crash_report,
)


class ExperimentWorker(QThread):
    """实验后台工作线程"""

    # 实时状态信号
    status_updated = pyqtSignal(str)
    metric_updated = pyqtSignal(str, float, int)  # 指标名, 值, step
    progress_updated = pyqtSignal(int, int)        # 当前步, 总步数
    log_updated = pyqtSignal(str)
    phase_updated = pyqtSignal(str, int, int)      # 阶段名, 当前进度, 总进度
    sample_ready = pyqtSignal(object)              # 逐样本识别结果

    # 阶段性信号
    data_generated = pyqtSignal(int)
    baseline_ready = pyqtSignal(object)
    epoch_ended = pyqtSignal(int, object)
    prune_stats_ready = pyqtSignal(object)
    importance_distribution = pyqtSignal(object)

    # 结束信号
    experiment_completed = pyqtSignal(object)
    experiment_error = pyqtSignal(str)

    def __init__(self, exp_type: str, config: dict):
        super().__init__()
        self.exp_type = exp_type
        self.config = config
        self.db = DBManager()
        self._is_running = True
        self._file_logger = None
        self._log_path = ""
        self._metrics = []
        self._last_status = None

    def run(self):
        """启动实验"""
        exp_id = self.config.get("exp_id", 0)
        # ---- 初始化文件日志 ----
        self._file_logger, self._log_path = setup_experiment_logger(exp_id)
        self._file_logger.info(f"实验类型: {self.exp_type}")
        self._file_logger.info(f"配置: {self._safe_config_str()}")
        self._ui_log(f"📁 日志文件: {self._log_path}")
        if exp_id:
            self.db.update_experiment_status(exp_id, "running")

        try:
            self.metric_updated.connect(self._accumulate_metric)
            self.experiment_completed.connect(self._persist_completed)
            self._ui_log(f"实验 #{exp_id} 开始执行")
            self._file_logger.info("实验开始执行")
            self.status_updated.emit("实验初始化...")

            # ---- 显存检查 ----
            if self._has_cuda():
                free, total = self._get_cuda_memory()
                self._file_logger.info(f"初始显存: free={free:.1f}GB / total={total:.1f}GB")
                if free < 4:
                    self._file_logger.warning(f"显存不足: 仅剩 {free:.1f}GB，建议关闭其他程序")
                    self._ui_log(f"⚠️ 警告: 显存仅剩 {free:.1f}GB，可能不够")

            if self.exp_type == "distillation":
                engine = DistillationEngine(self.config, self)
                engine._file_logger = self._file_logger  # 注入文件日志
                engine.run(
                    self.config.get("teacher_model_id"),
                    self.config.get("student_model_id")
                )
            elif self.exp_type == "pruning":
                engine = PruningEngine(self.config, self)
                engine._file_logger = self._file_logger
                engine.run(self.config.get("target_model_id"))

        except UserStopped:
            # 用户主动停止：不算崩溃，直接标记失败并通知 UI
            msg = "实验已停止（用户请求）"
            self._ui_log(f"⏹ {msg}")
            self.experiment_error.emit(msg)
            if exp_id:
                self.db.update_experiment_status(exp_id, "failed", "用户主动停止")
            self._last_status = "failed"
        except Exception as e:
            self._handle_error(e, exp_id)
            self._last_status = "failed"

    # ---- 结果持久化（worker 内完成，供单次/批量共用） ----

    def _accumulate_metric(self, name, value, step):
        if name in ("loss", "recovery_loss"):
            self._metrics.append({"epoch": step, "loss": value})

    def _persist_completed(self, results):
        """实验完成后写入数据库（页面不再重复写）"""
        exp_id = self.config.get("exp_id", 0)
        if not exp_id:
            return
        try:
            self.db.update_experiment_metrics(exp_id, self._metrics)
            self.db.update_experiment_results(
                exp_id, results,
                output_path=results.get("output_path", ""),
                report_path=results.get("report_path", ""),
            )
            self.db.update_experiment_status(exp_id, "completed")
            self._last_status = "completed"
        except Exception as e:
            if self._file_logger:
                self._file_logger.error(f"结果落库失败: {e}")

    def _safe_config_str(self):
        """安全的 config 字符串（避免打印过长路径）"""
        safe = dict(self.config)
        for k in ("teacher_path", "student_path", "image_dir", "seed_questions", "eval_questions"):
            if k in safe and isinstance(safe[k], str) and len(safe[k]) > 60:
                safe[k] = safe[k][:60] + "..."
        return str({k: v for k, v in safe.items() if k != "seed_questions"})

    # ---- 日志工具 ----

    def _ui_log(self, msg):
        """同时输出到 UI 和文件日志"""
        self.log_updated.emit(msg)
        if self._file_logger:
            # 去掉 Emoji 以保持日志文件干净
            clean = msg.encode("ascii", "ignore").decode().strip()
            if clean:
                self._file_logger.info(clean)

    def log(self, msg):
        """供外部引擎调用的日志方法（写入文件 + UI）"""
        self._ui_log(msg)

    def info(self, msg):
        self._ui_log(msg)

    def warning(self, msg):
        self._ui_log(f"⚠️ {msg}")

    def error(self, msg):
        self._ui_log(f"❌ {msg}")

    def _handle_error(self, exc, exp_id):
        """处理错误并记录到文件"""
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        tb = traceback.format_exc()
        full_msg = f"{error_msg}\n{tb}"

        # 写入文件日志
        if self._file_logger:
            self._file_logger.error(f"实验出错: {error_msg}")
            self._file_logger.error(tb)

        # 写出崩溃报告
        write_crash_report(full_msg)

        # 通知 UI
        self.log_updated.emit(f"❌ 实验出错: {error_msg}")
        self.log_updated.emit(f"📄 完整日志: {self._log_path}")
        self.experiment_error.emit(error_msg)

        if exp_id:
            self.db.update_experiment_status(exp_id, "failed", error_msg)

    # ---- CUDA 工具 ----

    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _get_cuda_memory(self):
        """获取显存状态 (free_gb, total_gb)"""
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated() / 1024**3
                return total - allocated, total
        except Exception:
            pass
        return 0, 0

    # ---- 停止支持（协作式取消） ----

    def is_cancelled(self) -> bool:
        """是否收到停止请求（供引擎在循环中轮询）"""
        return self.isInterruptionRequested() or not self._is_running

    def check_cancelled(self):
        """引擎在每个耗时步骤调用；收到停止请求时抛出 UserStopped"""
        if self.is_cancelled():
            raise UserStopped()

    def emit_phase(self, phase: str, current: int = 0, total: int = 0):
        """上报当前阶段与进度（供 UI 进度条/状态展示）"""
        self.phase_updated.emit(phase, current, total)

    def emit_sample(self, sample: dict):
        """上报一条逐样本识别结果（图片/问题/回答预览）"""
        self.sample_ready.emit(sample)

    def stop(self):
        """请求停止实验（协作式，引擎在下一个循环边界退出）"""
        self._is_running = False
        self.requestInterruption()
        msg = "收到停止请求，正在等待当前步骤完成..."
        self.log_updated.emit(msg)
        if self._file_logger:
            self._file_logger.warning(msg)

    def persist_distillation_data(self, items):
        """把蒸馏数据写入数据库（供引擎在生成完成后调用）"""
        exp_id = self.config.get("exp_id", 0)
        if not exp_id or not items:
            return
        try:
            self.db.add_distillation_data(exp_id, items)
            self._file_logger.info(f"已写入蒸馏数据: {len(items)} 条")
        except Exception as e:
            if self._file_logger:
                self._file_logger.warning(f"蒸馏数据落库失败: {e}")


class BatchWorker(QThread):
    """批量实验工作线程 - 按模板顺序执行多个蒸馏实验"""

    batch_progress = pyqtSignal(int, int, str)   # 当前序号, 总数, 实验名
    batch_finished = pyqtSignal(int, int)        # 成功数, 失败数
    log_updated = pyqtSignal(str)
    phase_updated = pyqtSignal(str, int, int)
    sample_ready = pyqtSignal(object)
    metric_updated = pyqtSignal(str, float, int)

    def __init__(self, configs: list):
        """configs: 已含 exp_id 与模型路径的完整配置列表"""
        super().__init__()
        self._configs = configs
        self._is_running = True
        self.db = DBManager()

    def run(self):
        ok_count, fail_count = 0, 0
        total = len(self._configs)
        for i, cfg in enumerate(self._configs):
            if not self._is_running:
                break
            exp_id = cfg.get("exp_id", 0)
            name = cfg.get("_batch_name", f"实验 #{exp_id}")
            self.batch_progress.emit(i + 1, total, name)
            self.log_updated.emit(f"═══ 开始 [{i + 1}/{total}] {name} ═══")

            worker = ExperimentWorker("distillation", cfg)
            worker.phase_updated.connect(self.phase_updated)
            worker.sample_ready.connect(self.sample_ready)
            worker.metric_updated.connect(self.metric_updated)
            worker.log_updated.connect(
                lambda msg, n=name: self.log_updated.emit(f"[{n}] {msg}")
            )
            worker.run()  # 同步执行（本线程内）

            status = worker._last_status or \
                (self.db.get_experiment_by_id(exp_id) or {}).get("status", "failed")
            if status == "completed":
                ok_count += 1
                self.log_updated.emit(f"✔ [{i + 1}/{total}] {name} 完成")
            else:
                fail_count += 1
                self.log_updated.emit(f"✘ [{i + 1}/{total}] {name} 失败")

        self.batch_finished.emit(ok_count, fail_count)
        self._write_batch_summary()

    def stop(self):
        self._is_running = False

    def _write_batch_summary(self):
        """批量结束后生成汇总报告（同图/同题不同参数横评入口）"""
        try:
            import datetime
            import json
            lines = [
                "# 批量实验汇总",
                "",
                f"- 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- 实验数量: {len(self._configs)}",
                "",
            ]
            for cfg in self._configs:
                exp_id = cfg.get("exp_id", 0)
                row = self.db.get_experiment_by_id(exp_id) or {}
                name = cfg.get("_batch_name", f"实验 #{exp_id}")
                status = row.get("status", "?")
                lines.append(f"## {name} [{status}]")
                lines.append(f"- 输出目录: {row.get('output_path', '-') or '-'}")
                try:
                    ev = json.loads(row.get("eval_results", "{}"))
                    scores = ev.get("scores", {})
                    dims = scores.get("dimensions", [])
                    b = scores.get("baseline", [])
                    a = scores.get("after", [])
                    if dims:
                        parts = []
                        for i, d in enumerate(dims):
                            bv = b[i] if i < len(b) else 0
                            av = a[i] if i < len(a) else 0
                            arrow = "↑" if av > bv + 1 else ("↓" if av < bv - 1 else "→")
                            parts.append(f"{d}: {bv:.1f}→{av:.1f}{arrow}")
                        lines.append("- 评分: " + " | ".join(parts))
                except (json.JSONDecodeError, TypeError):
                    pass
                lines.append("")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(EXPERIMENT_DIR / f"batch_summary_{ts}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.log_updated.emit(f"📄 批量汇总报告: {path}")
        except Exception as e:
            self.log_updated.emit(f"批量汇总生成失败: {e}")

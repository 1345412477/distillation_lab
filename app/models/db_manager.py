"""SQLite 数据库管理 - 实验记录与蒸馏数据持久化"""
import sqlite3
import json
import datetime
import threading
from typing import Optional
from ..utils.config import DB_PATH


class DBManager:
    """数据库管理器，单例模式"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._db_lock = threading.Lock()
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        with self._db_lock:
            cursor = self.conn.cursor()
            # 模型注册表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    repo_id TEXT DEFAULT '',
                    local_path TEXT DEFAULT '',
                    architecture TEXT DEFAULT '',
                    parameters REAL DEFAULT 0,
                    num_layers INTEGER DEFAULT 0,
                    hidden_size INTEGER DEFAULT 0,
                    dtype TEXT DEFAULT '',
                    file_size TEXT DEFAULT '',
                    created_at TEXT DEFAULT '',
                    baseline_results TEXT DEFAULT '{}'
                )
            """)
            # 实验记录
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    teacher_model_id INTEGER DEFAULT 0,
                    student_model_id INTEGER DEFAULT 0,
                    config TEXT DEFAULT '{}',
                    metrics TEXT DEFAULT '[]',
                    eval_results TEXT DEFAULT '{}',
                    compare_results TEXT DEFAULT '{}',
                    output_path TEXT DEFAULT '',
                    started_at TEXT DEFAULT '',
                    completed_at TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    report_path TEXT DEFAULT ''
                )
            """)
            # 蒸馏数据
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS distillation_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER DEFAULT 0,
                    instruction TEXT DEFAULT '',
                    output TEXT DEFAULT '',
                    created_at TEXT DEFAULT '',
                    source TEXT DEFAULT 'teacher_generated'
                )
            """)
            self.conn.commit()

    # ===== 模型操作 =====
    def add_model(self, name, source, repo_id="", local_path="",
                  architecture="", parameters=0, num_layers=0, hidden_size=0,
                  dtype="", file_size=""):
        with self._db_lock:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO model_registry
                (name, source, repo_id, local_path, architecture, parameters,
                 num_layers, hidden_size, dtype, file_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, source, repo_id, local_path, architecture, parameters,
                  num_layers, hidden_size, dtype, file_size, now))
            self.conn.commit()
            return cursor.lastrowid

    def get_all_models(self):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM model_registry ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_model_by_id(self, model_id):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE id=?", (model_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_model(self, model_id):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM model_registry WHERE id=?", (model_id,))
            self.conn.commit()

    def update_model_baseline(self, model_id, baseline_results):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE model_registry SET baseline_results=? WHERE id=?",
                           (json.dumps(baseline_results, ensure_ascii=False), model_id))
            self.conn.commit()

    # ===== 实验操作 =====
    def add_experiment(self, name, exp_type, config, teacher_model_id=0, student_model_id=0):
        with self._db_lock:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO experiment_records
                (name, type, status, teacher_model_id, student_model_id, config, started_at)
                VALUES (?, ?, 'pending', ?, ?, ?, ?)
            """, (name, exp_type, teacher_model_id, student_model_id,
                  json.dumps(config, ensure_ascii=False), now))
            self.conn.commit()
            return cursor.lastrowid

    def get_all_experiments(self):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM experiment_records ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def reset_stale_running(self):
        """把上次异常退出遗留的 running 实验标记为 failed（启动时调用一次）"""
        with self._db_lock:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE experiment_records
                SET status='failed', completed_at=?, error_message=?
                WHERE status='running'
            """, (now, "应用异常退出，实验被中断"))
            self.conn.commit()
            return cursor.rowcount

    def get_experiment_by_id(self, exp_id):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM experiment_records WHERE id=?", (exp_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_experiment_status(self, exp_id, status, error_message=""):
        with self._db_lock:
            cursor = self.conn.cursor()
            if status in ("completed", "failed"):
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("""
                    UPDATE experiment_records SET status=?, completed_at=?, error_message=?
                    WHERE id=?
                """, (status, now, error_message, exp_id))
            else:
                cursor.execute("UPDATE experiment_records SET status=? WHERE id=?",
                               (status, exp_id))
            self.conn.commit()

    def update_experiment_metrics(self, exp_id, metrics):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE experiment_records SET metrics=? WHERE id=?",
                           (json.dumps(metrics, ensure_ascii=False), exp_id))
            self.conn.commit()

    def update_experiment_results(self, exp_id, eval_results, compare_results=None,
                                  output_path="", report_path=""):
        with self._db_lock:
            cursor = self.conn.cursor()
            if report_path:
                cursor.execute("""
                    UPDATE experiment_records SET eval_results=?, output_path=?, report_path=?
                    WHERE id=?
                """, (json.dumps(eval_results, ensure_ascii=False), output_path, report_path, exp_id))
            else:
                cursor.execute("""
                    UPDATE experiment_records SET eval_results=?, output_path=?
                    WHERE id=?
                """, (json.dumps(eval_results, ensure_ascii=False), output_path, exp_id))
            if compare_results:
                cursor.execute("UPDATE experiment_records SET compare_results=? WHERE id=?",
                               (json.dumps(compare_results, ensure_ascii=False), exp_id))
            self.conn.commit()

    def delete_experiment(self, exp_id):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM experiment_records WHERE id=?", (exp_id,))
            cursor.execute("DELETE FROM distillation_data WHERE experiment_id=?", (exp_id,))
            self.conn.commit()

    # ===== 蒸馏数据操作 =====
    def add_distillation_data(self, experiment_id, items):
        with self._db_lock:
            cursor = self.conn.cursor()
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for item in items:
                cursor.execute("""
                    INSERT INTO distillation_data (experiment_id, instruction, output, created_at, source)
                    VALUES (?, ?, ?, ?, ?)
                """, (experiment_id,
                      item.get("instruction", item.get("question", "")),
                      item.get("output", item.get("answer", "")),
                      now, "teacher_generated"))
            self.conn.commit()

    def get_distillation_data(self, experiment_id):
        with self._db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM distillation_data WHERE experiment_id=? ORDER BY id",
                           (experiment_id,))
            return [dict(row) for row in cursor.fetchall()]

"""蒸馏引擎 - 封装离线数据蒸馏 + LoRA 训练逻辑（支持纯文本与多模态）"""
import json
import os
import torch
from ..utils.config import EXPERIMENT_DIR
from ..engines.evaluation import EvaluationEngine
from ..models.model_manager import resolve_model_path


def _get_multimodal_model_class():
    """自动获取当前 transformers 版本支持的多模态模型类"""
    # transformers 5.x 使用 AutoModelForImageTextToText
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except ImportError:
        pass
    # transformers 4.45+ 使用 AutoModelForVision2Seq
    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except ImportError:
        pass
    # 回退
    try:
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM
    except ImportError:
        pass
    raise ImportError(
        "当前 transformers 版本不支持多模态模型。请升级: pip install --upgrade transformers"
    )


class DistillationEngine:
    """蒸馏实验引擎 - 教师生成数据 → 学生微调 → 评估"""

    def __init__(self, config: dict, worker):
        self.config = config
        self.worker = worker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._file_logger = getattr(worker, "_file_logger", None)
        self._oom_safe = False  # 标记是否处于 OOM 风险阶段

    def _log(self, msg: str):
        """写入文件日志"""
        if self._file_logger:
            self._file_logger.info(msg)

    def _log_cuda_memory(self, tag: str = ""):
        """记录显存状态"""
        if self.device != "cuda":
            return
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - allocated
            self._log(f"[显存] {tag} - free={free:.2f}GB "
                      f"alloc={allocated:.2f}GB reserved={reserved:.2f}GB")
            if free < 2:
                self.worker.log_updated.emit(
                    f"⚠️ 显存不足: 仅剩 {free:.1f}GB，即将 OOM！"
                )
        except Exception:
            pass

    def _get_quantization_config(self):
        """如果启用 4-bit 量化，返回 BitsAndBytesConfig"""
        if not self.config.get("use_4bit", False):
            return None, {}
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = (
                torch.bfloat16
                if self.config.get("compute_dtype", "fp16") == "bf16"
                else torch.float16
            )
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(self.config.get("double_quant", True)),
                bnb_4bit_quant_type=self.config.get("quant_type", "nf4"),
            )
            kwargs = {
                "quantization_config": qconfig,
                "device_map": "auto",
            }
            self._log(
                f"启用 4-bit 量化加载模型（{self.config.get('quant_type', 'nf4')}, "
                f"double_quant={self.config.get('double_quant', True)}）"
            )
            return qconfig, kwargs
        except ImportError:
            self.worker.log_updated.emit("⚠️ bitsandbytes 未安装，无法使用 4-bit 量化")
            return None, {}

    def _model_kwargs(self, quant_kwargs: dict) -> dict:
        """构建模型加载参数（避免 device_map 重复）"""
        kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        # 量化时由 quant_kwargs 提供 device_map="auto"
        if not quant_kwargs:
            kwargs["device_map"] = self.device if self.device == "cuda" else None
        # 多模态模型在有 CUDA 时使用 sdpa
        if self.device == "cuda":
            kwargs["attn_implementation"] = "sdpa"
        return kwargs

    # ---- 可复用的模板/生成/优化器辅助 ----

    def _apply_chat_template(self, tokenizer, messages, add_generation_prompt=False):
        """应用聊天模板，兼容 Qwen 思考模式（enable_thinking）"""
        if self.config.get("enable_think", True):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    chat_template_kwargs={"enable_thinking": True},
                )
            except TypeError:
                pass
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=True,
                )
            except TypeError:
                pass
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _generation_kwargs(self):
        """教师生成参数（temperature/top_p/top_k/beam/惩罚 全覆盖）"""
        kwargs = {
            "max_new_tokens": self.config.get("max_new_tokens", 256),
            "do_sample": True,
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 0.9),
            "repetition_penalty": self.config.get("repetition_penalty", 1.0),
        }
        top_k = int(self.config.get("top_k", 0))
        if top_k > 0:
            kwargs["top_k"] = top_k
        num_beams = int(self.config.get("num_beams", 1))
        if num_beams > 1:
            kwargs["num_beams"] = num_beams
            kwargs["do_sample"] = False
        return kwargs

    def _build_optimizer(self, params):
        """按配置构建优化器（adamw / adamw8bit / sgd）"""
        name = self.config.get("optimizer", "adamw")
        lr = self.config.get("learning_rate", 5e-5)
        wd = self.config.get("weight_decay", 0.01)
        if name == "adamw8bit":
            try:
                from bitsandbytes.optim import AdamW8bit
                return AdamW8bit(params, lr=lr, weight_decay=wd)
            except ImportError:
                self.worker.log_updated.emit("bitsandbytes 未安装，回退到 AdamW")
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def _build_scheduler(self, optimizer, total_steps):
        """按配置构建学习率调度器（warmup 线性）"""
        warmup = int(self.config.get("warmup_steps", 0))
        if warmup <= 0:
            return None

        def lr_lambda(step):
            return min(1.0, (step + 1) / warmup)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ================================================================
    # 入口：自动判断走纯文本还是多模态路径
    # ================================================================
    def run(self, teacher_model_id, student_model_id):
        # 固定随机种子，保证实验可复现
        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 解析 ModelScope 路径 → 实际 snapshot 目录
        teacher_path = resolve_model_path(self.config.get("teacher_path", ""))
        student_path = resolve_model_path(self.config.get("student_path", ""))
        self.config["teacher_path"] = teacher_path
        self.config["student_path"] = student_path

        image_dir = self.config.get("image_dir", "").strip()
        if image_dir and os.path.isdir(image_dir):
            # 校验教师和学生是否都是多模态模型
            teacher_is_vl = self._is_multimodal_model(teacher_path)
            student_is_vl = self._is_multimodal_model(student_path)
            if not teacher_is_vl and not student_is_vl:
                self.worker.log_updated.emit(
                    "⚠️ 教师和学生模型都不是多模态模型，忽略图像目录，按纯文本执行"
                )
                self._run_text(teacher_model_id, student_model_id)
            elif not teacher_is_vl:
                self.worker.log_updated.emit("⚠️ 教师不是多模态模型，无法生成图像描述，按纯文本执行")
                self._run_text(teacher_model_id, student_model_id)
            elif not student_is_vl:
                self.worker.log_updated.emit(
                    "⚠️ 学生不是多模态模型，但教师将生成图像描述数据供学生学习文本输出能力"
                )
                self._run_multimodal(teacher_model_id, student_model_id)
            else:
                self.worker.log_updated.emit("检测到图像数据集 → 执行多模态蒸馏")
                self._run_multimodal(teacher_model_id, student_model_id)
        else:
            self._run_text(teacher_model_id, student_model_id)

    # ================================================================
    # 工具方法
    # ================================================================
    def _is_multimodal_model(self, model_path: str) -> bool:
        """通过 config 判断模型是否为多模态（是否包含 vision_config）"""
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            return hasattr(cfg, "vision_config") or hasattr(cfg, "vision_encoder")
        except Exception:
            return False

    def _scan_images(self, image_dir: str, max_images: int = 50):
        """扫描目录中的图片并返回绝对路径列表"""
        from PIL import Image
        valid_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        images = []
        for fname in sorted(os.listdir(image_dir)):
            if len(images) >= max_images:
                break
            fpath = os.path.join(image_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(valid_ext):
                # 尝试用 Pillow 验证（imghdr 在 Python 3.13 已被移除）
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    images.append(fpath)
                except Exception:
                    self.worker.log_updated.emit(
                        f"[跳过] 无法识别的图片文件: {os.path.basename(fpath)}"
                    )
        return images

    # ================================================================
    # 纯文本蒸馏路径（原逻辑）
    # ================================================================
    def _run_text(self, teacher_model_id, student_model_id):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ---- 0. 量化配置 ----
        _, quant_kwargs = self._get_quantization_config()
        eval_qs = self.config.get("eval_questions", [
            "什么是深度学习？", "用Python写一个计算斐波那契数列的函数。", "HTTP和HTTPS有什么区别？"
        ])
        eval_max_tokens = self.config.get("eval_max_new_tokens", 64)

        # ---- 1. 加载教师 ----
        self.worker.status_updated.emit("正在加载教师模型...")
        teacher_path = self.config.get("teacher_path", "")
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            **self._model_kwargs(quant_kwargs),
            **quant_kwargs,
        )
        if self.device != "cuda" and not quant_kwargs:
            teacher_model.to(self.device)
        teacher_model.eval()
        self._log_cuda_memory("教师模型加载后")
        self.worker.status_updated.emit("教师模型加载完成")

        # ---- 2. 种子问题 ----
        seed_questions = list(self.config.get("seed_questions", []))
        num_generate = self.config.get("num_generate", 50)
        extra_topics = [
            "解释什么是数据库索引", "什么是微服务架构", "TCP三次握手的过程",
            "什么是Docker容器", "解释MapReduce的原理", "什么是消息队列",
            "Git中rebase和merge的区别", "什么是持续集成", "解释什么是负载均衡",
            "什么是缓存穿透", "解释什么是设计模式", "什么是NoSQL数据库",
            "写一个Python冒泡排序", "用Python实现栈和队列", "写一个Python装饰器示例",
            "Python中生成器和迭代器的区别", "什么是Python的GIL",
            "如何用Python发送HTTP请求", "写一个Python正则表达式匹配邮箱",
            "什么是时间复杂度", "解释空间复杂度", "什么是动态规划",
            "解释什么是二叉搜索树", "什么是哈希表", "解释图的最短路径算法",
            "什么是机器学习中的交叉验证", "解释什么是注意力机制", "什么是Transformer架构",
            "解释什么是迁移学习", "什么是数据增强", "解释什么是词嵌入",
        ]
        while len(seed_questions) < num_generate:
            seed_questions.extend(extra_topics)
        seed_questions = seed_questions[:num_generate]

        # ---- 3. 教师生成 ----
        self.worker.status_updated.emit(f"教师模型生成 {len(seed_questions)} 条数据...")
        self.worker.emit_phase("数据生成", 0, len(seed_questions))
        distillation_data = []
        for i, question in enumerate(seed_questions):
            self.worker.check_cancelled()
            messages = [{"role": "user", "content": question}]
            text = self._apply_chat_template(teacher_tokenizer, messages, add_generation_prompt=True)
            inputs = teacher_tokenizer(text, return_tensors="pt").to(teacher_model.device)
            with torch.no_grad():
                outputs = teacher_model.generate(**inputs, **self._generation_kwargs())
            response = teacher_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            distillation_data.append({"instruction": question, "output": response})
            self.worker.progress_updated.emit(i + 1, len(seed_questions))
            self.worker.emit_phase("数据生成", i + 1, len(seed_questions))
            self.worker.emit_sample({
                "type": "generated",
                "question": question,
                "answer": response[:200],
            })

        self.worker.data_generated.emit(len(distillation_data))
        self.worker.status_updated.emit(f"蒸馏数据生成完成: {len(distillation_data)} 条")
        self.worker.persist_distillation_data(distillation_data)

        # ---- 3b. 教师评估（同一评估集，用于三模型对比） ----
        self.worker.emit_phase("教师评估", 0, len(eval_qs))
        teacher_results = self._eval(
            teacher_model, teacher_tokenizer, eval_qs,
            max_new_tokens=eval_max_tokens,
        )
        self.worker.log_updated.emit("教师评估完成")
        self.worker.emit_phase("教师评估", len(eval_qs), len(eval_qs))

        del teacher_model, teacher_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_cuda_memory("教师释放后")

        # ---- 4. 加载学生 ----
        self.worker.status_updated.emit("正在加载学生模型...")
        student_path = self.config.get("student_path", "")
        student_tokenizer = AutoTokenizer.from_pretrained(student_path)
        student_model = AutoModelForCausalLM.from_pretrained(
            student_path,
            **self._model_kwargs(quant_kwargs),
            **quant_kwargs,
        )
        if self.device != "cuda" and not quant_kwargs:
            student_model.to(self.device)
        if self.device == "cuda":
            student_model.gradient_checkpointing_enable()
        self.worker.log_updated.emit(
            f"学生模型参数量: {sum(p.numel() for p in student_model.parameters())/1e9:.3f}B"
        )
        self._log_cuda_memory("学生模型加载后")

        # ---- 5. 基线评估 ----
        self.worker.status_updated.emit("正在进行蒸馏前基线评估...")
        self.worker.emit_phase("基线评估", 0, len(eval_qs))
        baseline_results = self._eval(
            student_model, student_tokenizer, eval_qs,
            max_new_tokens=eval_max_tokens,
        )
        self.worker.baseline_ready.emit({"type": "baseline", "results": baseline_results})
        self.worker.log_updated.emit("基线评估完成")
        self.worker.emit_phase("基线评估", len(eval_qs), len(eval_qs))

        # ---- 6. LoRA 蒸馏训练 ----
        self.worker.status_updated.emit("准备 LoRA 蒸馏训练...")
        train_texts = []
        for item in distillation_data:
            msgs = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]},
            ]
            train_texts.append(student_tokenizer.apply_chat_template(msgs, tokenize=False))
        if student_tokenizer.pad_token is None:
            student_tokenizer.pad_token = student_tokenizer.eos_token

        from peft import get_peft_model, LoraConfig

        # 4-bit 量化模型需要 prepare_for_kbit_training
        if quant_kwargs:
            from peft import prepare_model_for_kbit_training
            student_model = prepare_model_for_kbit_training(student_model)
            self.worker.log_updated.emit("已应用 prepare_model_for_kbit_training")

        lora_config = LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        student_model = get_peft_model(student_model, lora_config)
        self.worker.log_updated.emit(
            f"LoRA 可训练参数: {sum(p.numel() for p in student_model.parameters() if p.requires_grad)/1e6:.1f}M"
        )
        student_model.train()
        optimizer = self._build_optimizer(student_model.parameters())

        max_seq_len = self.config.get("max_seq_len", 512)
        all_input_ids, all_labels, all_masks = [], [], []
        for text in train_texts:
            tokens = student_tokenizer(
                text, max_length=max_seq_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            ids = tokens["input_ids"].squeeze()
            mask = tokens["attention_mask"].squeeze()
            labels = ids.clone()
            labels[mask == 0] = -100
            all_input_ids.append(ids)
            all_labels.append(labels)
            all_masks.append(mask)

        epochs = self.config.get("epochs", 3)
        batch_size = self.config.get("batch_size", 2)
        grad_clip = self.config.get("grad_clip", 1.0)
        grad_accum = max(1, int(self.config.get("grad_accum_steps", 1)))
        num_batches = (len(all_input_ids) + batch_size - 1) // batch_size
        total_steps = epochs * ((num_batches + grad_accum - 1) // grad_accum)
        scheduler = self._build_scheduler(optimizer, total_steps)
        metrics = []
        import time
        train_start = time.time()
        for epoch in range(epochs):
            self.worker.check_cancelled()
            self._log_cuda_memory(f"Epoch {epoch+1}/{epochs} 开始")
            epoch_loss, nb = 0.0, 0
            opt_accum = 0
            for i in range(0, len(all_input_ids), batch_size):
                self.worker.check_cancelled()
                self.worker.emit_phase(
                    "训练", epoch * num_batches + i // batch_size, epochs * num_batches
                )
                bid = torch.stack(all_input_ids[i:i+batch_size]).to(student_model.device)
                blab = torch.stack(all_labels[i:i+batch_size]).to(student_model.device)
                bmask = torch.stack(all_masks[i:i+batch_size]).to(student_model.device)
                outputs = student_model(input_ids=bid, labels=blab, attention_mask=bmask)
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.worker.log_updated.emit("[警告] 检测到 NaN/Inf loss，跳过此 batch")
                    optimizer.zero_grad()
                    continue
                loss = loss / grad_accum
                loss.backward()
                opt_accum += 1
                if opt_accum % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    opt_accum = 0
                loss_val = loss.item()
                epoch_loss += loss_val
                nb += 1
                step = epoch * len(all_input_ids) // batch_size + i // batch_size
                self.worker.metric_updated.emit("loss", loss_val, step)
                elapsed = time.time() - train_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                self.worker.metric_updated.emit("speed", round(speed, 3), step)
                self.worker.log_updated.emit(
                    f"Epoch {epoch+1}/{epochs} - batch {i//batch_size+1}/{(len(all_input_ids)-1)//batch_size+1} - Loss: {loss_val:.4f}"
                )
            if opt_accum % grad_accum != 0 and opt_accum > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            avg_loss = epoch_loss / max(1, nb)
            metrics.append({"epoch": epoch + 1, "avg_loss": avg_loss})
            self.worker.epoch_ended.emit(epoch + 1, {"loss": avg_loss})
            self.worker.status_updated.emit(f"Epoch {epoch+1}/{epochs} 完成 - Loss: {avg_loss:.4f}")
        self.worker.log_updated.emit("蒸馏训练完成")
        student_model = student_model.merge_and_unload()

        # ---- 7. 评估 ----
        self.worker.status_updated.emit("正在进行蒸馏后评估...")
        self.worker.emit_phase("蒸馏后评估", 0, len(eval_qs))
        after_results = self._eval(
            student_model, student_tokenizer, eval_qs,
            max_new_tokens=eval_max_tokens,
        )
        self.worker.log_updated.emit("蒸馏后评估完成")
        self.worker.emit_phase("蒸馏后评估", len(eval_qs), len(eval_qs))

        # ---- 8. 评分与报告 ----
        references = self.config.get("reference_answers") or []
        scores = EvaluationEngine.compute_scores(baseline_results, after_results, references)
        t_scores = EvaluationEngine.compute_scores(teacher_results, teacher_results, references)
        scores["teacher"] = t_scores["baseline"]

        # ---- 9. 保存 ----
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(EXPERIMENT_DIR / f"distill_{ts}")
        student_model.save_pretrained(output_path)
        student_tokenizer.save_pretrained(output_path)
        with open(os.path.join(output_path, "distillation_data.json"), "w", encoding="utf-8") as f:
            json.dump(distillation_data, f, ensure_ascii=False, indent=2)

        report = EvaluationEngine.generate_comparison_report(
            baseline_results, after_results, scores,
            teacher=teacher_results,
            model_name_a="基线", model_name_b="蒸馏后",
            extra_info={
                "type": "distillation(text)",
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_data": len(distillation_data),
            },
        )
        report_path = os.path.join(output_path, "evaluation_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        self.worker.experiment_completed.emit({
            "type": "distillation",
            "baseline": baseline_results,
            "after": after_results,
            "teacher": teacher_results,
            "scores": scores,
            "metrics": metrics,
            "output_path": output_path,
            "report_path": report_path,
            "num_data": len(distillation_data),
            "eval_questions": eval_qs,
        })

    def _eval(self, model, tokenizer, questions, max_new_tokens=64):
        """纯文本评估（统一入口，支持协作式取消）"""
        return EvaluationEngine.simple_eval(
            model, tokenizer, questions,
            device=str(self.device),
            max_new_tokens=max_new_tokens,
            cancel_check=self.worker.check_cancelled,
        )

    # ================================================================
    # 多模态蒸馏路径
    # ================================================================
    def _run_multimodal(self, teacher_model_id, student_model_id):
        from PIL import Image
        from transformers import AutoProcessor

        MultimodalModel = _get_multimodal_model_class()

        image_dir = self.config.get("image_dir", "")
        image_prompts = list(self.config.get("image_prompts") or [])
        if not image_prompts and self.config.get("image_prompt_template", "").strip():
            image_prompts = [self.config["image_prompt_template"]]
        if not image_prompts:
            image_prompts = ["请详细描述这张图片的内容"]
        max_images = self.config.get("max_images", 50)
        eval_qs = self.config.get("eval_questions", ["请描述这张图片中的主要内容"])
        eval_max_tokens = self.config.get("eval_max_new_tokens", 64)

        # ---- 0. 量化配置 ----
        _, quant_kwargs = self._get_quantization_config()

        # ---- 1. 扫描图像 ----
        self.worker.status_updated.emit("正在扫描图像数据集...")
        image_paths = self._scan_images(image_dir, max_images)
        if not image_paths:
            raise FileNotFoundError(f"图像目录中未找到有效图片: {image_dir}")
        self.worker.log_updated.emit(f"找到 {len(image_paths)} 张图片用于多模态蒸馏")

        # ---- 2. 加载多模态教师模型 ----
        self.worker.status_updated.emit("正在加载多模态教师模型...")
        teacher_path = self.config.get("teacher_path", "")
        teacher_processor = AutoProcessor.from_pretrained(teacher_path)
        teacher_model = MultimodalModel.from_pretrained(
            teacher_path,
            **self._model_kwargs(quant_kwargs),
            **quant_kwargs,
        )
        if self.device != "cuda" and not quant_kwargs:
            teacher_model.to(self.device)
        teacher_model.eval()
        self._log_cuda_memory("多模态教师模型加载后")
        self.worker.status_updated.emit("多模态教师模型加载完成")

        # ---- 3. 教师模型生成多模态蒸馏数据 ----
        total_generate = len(image_paths) * len(image_prompts)
        self.worker.status_updated.emit(
            f"教师模型生成多模态数据: {len(image_paths)} 张图 × {len(image_prompts)} 个提示词 = {total_generate} 条"
        )
        self.worker.emit_phase("数据生成", 0, total_generate)
        distillation_data = []
        idx = 0
        for i, img_path in enumerate(image_paths):
            self.worker.check_cancelled()
            try:
                image = Image.open(img_path).convert("RGB")
                for prompt in image_prompts:
                    self.worker.check_cancelled()
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    text = self._apply_chat_template(
                        teacher_processor, messages, add_generation_prompt=True
                    )
                    inputs = teacher_processor(
                        text=[text], images=[image], return_tensors="pt",
                    ).to(teacher_model.device)

                    with torch.no_grad():
                        outputs = teacher_model.generate(
                            **inputs, **self._generation_kwargs()
                        )
                    response = teacher_processor.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ).strip()

                    distillation_data.append({
                        "image_path": img_path,
                        "instruction": prompt,
                        "output": response,
                    })
                    idx += 1
                    self.worker.progress_updated.emit(idx, total_generate)
                    self.worker.emit_phase("数据生成", idx, total_generate)
                    self.worker.emit_sample({
                        "type": "generated",
                        "image": os.path.basename(img_path),
                        "image_path": img_path,
                        "question": prompt,
                        "answer": response[:200],
                    })
                    self.worker.log_updated.emit(
                        f"[多模态数据生成] {idx}/{total_generate}: "
                        f"{os.path.basename(img_path)} | {prompt[:24]}"
                    )
            except Exception as e:
                self.worker.log_updated.emit(
                    f"[跳过] 处理图片 {os.path.basename(img_path)} 失败: {e}"
                )

        if not distillation_data:
            raise RuntimeError("未生成任何有效多模态数据")

        self.worker.data_generated.emit(len(distillation_data))
        self.worker.status_updated.emit(f"多模态数据生成完成: {len(distillation_data)} 条")
        serializable_data = [
            {
                "image_path": d["image_path"],
                "instruction": d["instruction"],
                "output": d["output"],
            }
            for d in distillation_data
        ]
        self.worker.persist_distillation_data(serializable_data)

        # ---- 3b. 教师评估（同一评估集，用于三模型对比） ----
        eval_images = [d["image_path"] for d in distillation_data[:min(3, len(distillation_data))]]
        eval_total = len(eval_images) * len(eval_qs)
        self.worker.emit_phase("教师评估", 0, eval_total)
        teacher_results = self._simple_eval_multimodal(
            teacher_model, teacher_processor, eval_qs, eval_images,
            max_new_tokens=eval_max_tokens,
        )
        self.worker.log_updated.emit("教师评估完成")
        self.worker.emit_phase("教师评估", eval_total, eval_total)

        # 释放教师显存
        del teacher_model, teacher_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_cuda_memory("多模态教师释放后")

        # ---- 4. 加载学生模型（支持多模态或纯文本） ----
        self.worker.status_updated.emit("正在加载学生模型...")
        student_path = self.config.get("student_path", "")
        student_is_vl = self._is_multimodal_model(student_path)

        if student_is_vl:
            from transformers import AutoProcessor
            student_processor = AutoProcessor.from_pretrained(student_path)
            student_model = MultimodalModel.from_pretrained(
                student_path,
                **self._model_kwargs(quant_kwargs),
                **quant_kwargs,
            )
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            student_tokenizer = AutoTokenizer.from_pretrained(student_path)
            if student_tokenizer.pad_token is None:
                student_tokenizer.pad_token = student_tokenizer.eos_token
            student_model = AutoModelForCausalLM.from_pretrained(
                student_path,
                **self._model_kwargs(quant_kwargs),
                **quant_kwargs,
            )
            self.worker.log_updated.emit("学生模型为纯文本模型，将以文本蒸馏模式训练")
        if self.device != "cuda" and not quant_kwargs:
            student_model.to(self.device)
        if self.device == "cuda":
            student_model.gradient_checkpointing_enable()
        self.worker.log_updated.emit(
            f"多模态学生模型参数量: {sum(p.numel() for p in student_model.parameters())/1e9:.3f}B"
        )
        self._log_cuda_memory("多模态学生模型加载后")

        # ---- 5. 多模态基线评估 ----
        self.worker.status_updated.emit("正在进行蒸馏前基线评估...")
        self.worker.emit_phase("基线评估", 0, eval_total)

        if student_is_vl:
            baseline_results = self._simple_eval_multimodal(
                student_model, student_processor, eval_qs, eval_images,
                max_new_tokens=eval_max_tokens,
            )
        else:
            baseline_results = self._eval(
                student_model, student_tokenizer, eval_qs,
                max_new_tokens=eval_max_tokens,
            )
        self.worker.baseline_ready.emit({"type": "baseline", "results": baseline_results})
        self.worker.log_updated.emit("基线评估完成")
        self.worker.emit_phase("基线评估", eval_total, eval_total)

        # ---- 6. 冻结视觉编码器（多模态学生才需要） + LoRA 训练 ----
        self.worker.status_updated.emit("准备 LoRA 蒸馏训练...")

        # 冻结视觉编码器（仅多模态学生）
        if student_is_vl and hasattr(student_model, "vision_tower"):
            for param in student_model.vision_tower.parameters():
                param.requires_grad = False
            self.worker.log_updated.emit("已冻结 vision_tower")
        if student_is_vl and hasattr(student_model, "mm_projector"):
            for param in student_model.mm_projector.parameters():
                param.requires_grad = False
            self.worker.log_updated.emit("已冻结 mm_projector")

        # 4-bit 量化模型需要 prepare_for_kbit_training
        if quant_kwargs:
            from peft import prepare_model_for_kbit_training
            student_model = prepare_model_for_kbit_training(student_model)
            self.worker.log_updated.emit("已应用 prepare_model_for_kbit_training")

        # LoRA
        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        student_model = get_peft_model(student_model, lora_config)
        # 验证只有 LoRA 参数可训练
        trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student_model.parameters())
        self.worker.log_updated.emit(
            f"LoRA 可训练参数: {trainable/1e6:.1f}M / {total/1e6:.1f}M total ({trainable/total*100:.2f}%)"
        )
        student_model.train()
        optimizer = self._build_optimizer(student_model.parameters())

        # ---- 7. 蒸馏训练循环（根据学生类型自动选择图文或纯文本路径） ----
        epochs = self.config.get("epochs", 3)
        batch_size = self.config.get("batch_size", 2) if not student_is_vl else 1
        grad_clip = self.config.get("grad_clip", 1.0)
        grad_accum = max(1, int(self.config.get("grad_accum_steps", 1)))
        metrics = []

        self.worker.log_updated.emit(
            f"开始蒸馏训练: {epochs} epochs, batch_size={batch_size}"
            + (" (图文)" if student_is_vl else " (纯文本)")
        )

        # ---- 7a. 图文学生：使用 processor + 图像输入 ----
        if student_is_vl:
            num_batches = (len(distillation_data) + batch_size - 1) // batch_size
            scheduler = self._build_scheduler(optimizer, epochs * num_batches)
            import time
            train_start = time.time()
            for epoch in range(epochs):
                self.worker.check_cancelled()
                self._log_cuda_memory(f"多模态 Epoch {epoch+1}/{epochs} 开始")
                epoch_loss, nb = 0.0, 0
                opt_accum = 0
                for i in range(0, len(distillation_data), batch_size):
                    self.worker.check_cancelled()
                    self.worker.emit_phase(
                        "训练", epoch * num_batches + i // batch_size, epochs * num_batches
                    )
                    batch_items = distillation_data[i:i+batch_size]
                    images, texts = [], []
                    for item in batch_items:
                        try:
                            img = Image.open(item["image_path"]).convert("RGB")
                        except Exception as e:
                            continue
                        images.append(img)
                        msgs = [
                            {"role": "user", "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": item["instruction"]},
                            ]},
                            {"role": "assistant", "content": item["output"]},
                        ]
                        texts.append(student_processor.apply_chat_template(msgs, tokenize=False))
                    if not texts:
                        continue
                    inputs = student_processor(
                        text=texts, images=images, padding=True, return_tensors="pt",
                    ).to(student_model.device)
                    labels = inputs["input_ids"].clone()
                    labels[inputs["attention_mask"] == 0] = -100
                    # 将 processor 所有输出传给模型（含 image_grid_thw 等关键参数）
                    model_kwargs = dict(inputs)
                    model_kwargs["labels"] = labels
                    outputs = student_model(**model_kwargs)
                    loss = outputs.loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.worker.log_updated.emit("[警告] NaN/Inf loss，跳过此 batch")
                        optimizer.zero_grad()
                        continue
                    loss = loss / grad_accum
                    loss.backward()
                    opt_accum += 1
                    if opt_accum % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip)
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()
                        opt_accum = 0
                    loss_val = loss.item()
                    epoch_loss += loss_val
                    nb += 1
                    step = epoch * len(distillation_data) + i
                    self.worker.metric_updated.emit("loss", loss_val, step)
                    elapsed = time.time() - train_start
                    speed = (step + 1) / elapsed if elapsed > 0 else 0
                    self.worker.metric_updated.emit("speed", round(speed, 3), step)
                    self.worker.log_updated.emit(
                        f"Epoch {epoch+1}/{epochs} - batch {i+1}/{len(distillation_data)} - Loss: {loss_val:.4f}"
                    )
                if opt_accum % grad_accum != 0 and opt_accum > 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                avg_loss = epoch_loss / max(1, nb)
                metrics.append({"epoch": epoch + 1, "avg_loss": avg_loss})
                self.worker.epoch_ended.emit(epoch + 1, {"loss": avg_loss})
                self.worker.status_updated.emit(f"Epoch {epoch+1}/{epochs} 完成 - Loss: {avg_loss:.4f}")

        # ---- 7b. 纯文本学生：使用 tokenizer + 纯文本训练 ----
        else:
            train_texts = []
            for item in distillation_data:
                msgs = [
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["output"]},
                ]
                train_texts.append(student_tokenizer.apply_chat_template(msgs, tokenize=False))

            max_seq_len = self.config.get("max_seq_len", 512)
            all_input_ids, all_labels, all_masks = [], [], []
            for text in train_texts:
                tokens = student_tokenizer(
                    text, max_length=max_seq_len, truncation=True,
                    padding="max_length", return_tensors="pt",
                )
                ids = tokens["input_ids"].squeeze()
                mask = tokens["attention_mask"].squeeze()
                labels = ids.clone()
                labels[mask == 0] = -100
                all_input_ids.append(ids)
                all_labels.append(labels)
                all_masks.append(mask)

            num_batches = (len(all_input_ids) + batch_size - 1) // batch_size
            scheduler = self._build_scheduler(optimizer, epochs * num_batches)
            import time
            train_start = time.time()
            for epoch in range(epochs):
                self.worker.check_cancelled()
                epoch_loss, nb = 0.0, 0
                opt_accum = 0
                for i in range(0, len(all_input_ids), batch_size):
                    self.worker.check_cancelled()
                    self.worker.emit_phase(
                        "训练", epoch * num_batches + i // batch_size, epochs * num_batches
                    )
                    bid = torch.stack(all_input_ids[i:i+batch_size]).to(student_model.device)
                    blab = torch.stack(all_labels[i:i+batch_size]).to(student_model.device)
                    bmask = torch.stack(all_masks[i:i+batch_size]).to(student_model.device)
                    outputs = student_model(input_ids=bid, labels=blab, attention_mask=bmask)
                    loss = outputs.loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.worker.log_updated.emit("[警告] NaN/Inf loss，跳过此 batch")
                        optimizer.zero_grad()
                        continue
                    loss = loss / grad_accum
                    loss.backward()
                    opt_accum += 1
                    if opt_accum % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip)
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()
                        opt_accum = 0
                    loss_val = loss.item()
                    epoch_loss += loss_val
                    nb += 1
                    step = epoch * len(all_input_ids) // batch_size + i // batch_size
                    self.worker.metric_updated.emit("loss", loss_val, step)
                    elapsed = time.time() - train_start
                    speed = (step + 1) / elapsed if elapsed > 0 else 0
                    self.worker.metric_updated.emit("speed", round(speed, 3), step)
                    self.worker.log_updated.emit(
                        f"Epoch {epoch+1}/{epochs} - batch {i//batch_size+1}/{(len(all_input_ids)-1)//batch_size+1} - Loss: {loss_val:.4f}"
                    )
                if opt_accum % grad_accum != 0 and opt_accum > 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                avg_loss = epoch_loss / max(1, nb)
                metrics.append({"epoch": epoch + 1, "avg_loss": avg_loss})
                self.worker.epoch_ended.emit(epoch + 1, {"loss": avg_loss})
                self.worker.status_updated.emit(f"Epoch {epoch+1}/{epochs} 完成 - Loss: {avg_loss:.4f}")

        self.worker.log_updated.emit("蒸馏训练完成")
        student_model = student_model.merge_and_unload()

        # ---- 8. 蒸馏后评估 ----
        self.worker.status_updated.emit("正在进行蒸馏后评估...")
        self.worker.emit_phase("蒸馏后评估", 0, eval_total)
        if student_is_vl:
            after_results = self._simple_eval_multimodal(
                student_model, student_processor, eval_qs, eval_images,
                max_new_tokens=eval_max_tokens,
            )
        else:
            after_results = self._eval(
                student_model, student_tokenizer, eval_qs,
                max_new_tokens=eval_max_tokens,
            )
        self.worker.log_updated.emit("蒸馏后评估完成")
        self.worker.emit_phase("蒸馏后评估", eval_total, eval_total)

        # ---- 9. 评分与报告 ----
        references = self.config.get("reference_answers") or []
        scores = EvaluationEngine.compute_scores(baseline_results, after_results, references)
        t_scores = EvaluationEngine.compute_scores(teacher_results, teacher_results, references)
        scores["teacher"] = t_scores["baseline"]

        # 可选 CLIP 图像对齐评分（衡量描述是否与图片语义一致）
        if self.config.get("use_clip_score", False):
            self.worker.status_updated.emit("正在计算 CLIP 图像对齐评分...")
            self.worker.emit_phase("CLIP 评估", 0, 1)
            clip_items = lambda rs: [
                {"image_path": r.get("image_path"), "answer": r.get("answer")}
                for r in rs
            ]
            teacher_clip = EvaluationEngine.compute_clip_alignment(clip_items(teacher_results))
            baseline_clip = EvaluationEngine.compute_clip_alignment(clip_items(baseline_results))
            after_clip = EvaluationEngine.compute_clip_alignment(clip_items(after_results))
            if teacher_clip and baseline_clip and after_clip:
                scores["dimensions"] = scores["dimensions"] + ["图像对齐"]
                scores["baseline"] = scores["baseline"] + [
                    round(sum(baseline_clip) / len(baseline_clip), 1)
                ]
                scores["after"] = scores["after"] + [
                    round(sum(after_clip) / len(after_clip), 1)
                ]
                scores["teacher"] = scores["teacher"] + [
                    round(sum(teacher_clip) / len(teacher_clip), 1)
                ]
                self.worker.log_updated.emit(
                    "CLIP 图像对齐: "
                    f"教师 {scores['teacher'][-1]:.1f} / "
                    f"蒸馏前 {scores['baseline'][-1]:.1f} / "
                    f"蒸馏后 {scores['after'][-1]:.1f}"
                )
            else:
                self.worker.log_updated.emit(
                    "⚠️ CLIP 评分不可用（需网络下载模型或图片路径缺失），已跳过该维度"
                )
            self.worker.emit_phase("CLIP 评估", 1, 1)

        # ---- 10. 保存 ----
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(EXPERIMENT_DIR / f"multimodal_distill_{ts}")
        student_model.save_pretrained(output_path)
        if student_is_vl:
            student_processor.save_pretrained(output_path)
        else:
            student_tokenizer.save_pretrained(output_path)
        # 保存蒸馏数据（含图片路径映射）
        serializable_data = []
        for d in distillation_data:
            serializable_data.append({
                "image_path": d["image_path"],
                "instruction": d["instruction"],
                "output": d["output"],
            })
        with open(os.path.join(output_path, "distillation_data.json"), "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        report = EvaluationEngine.generate_comparison_report(
            baseline_results, after_results, scores,
            teacher=teacher_results,
            model_name_a="基线", model_name_b="蒸馏后",
            extra_info={
                "type": "distillation(multimodal)",
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_data": len(distillation_data),
            },
        )
        report_path = os.path.join(output_path, "evaluation_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        self.worker.experiment_completed.emit({
            "type": "distillation",
            "subtype": "multimodal",
            "baseline": baseline_results,
            "after": after_results,
            "teacher": teacher_results,
            "scores": scores,
            "metrics": metrics,
            "output_path": output_path,
            "report_path": report_path,
            "num_data": len(distillation_data),
            "num_images": len(image_paths),
            "eval_questions": eval_qs,
        })

    def _simple_eval_multimodal(self, model, processor, questions, image_paths, max_new_tokens=64):
        """多模态评估：对每张图片提问并收集回答"""
        from PIL import Image
        model.eval()
        results = []
        for img_path in image_paths:
            self.worker.check_cancelled()
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            for q in questions:
                self.worker.check_cancelled()
                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": q},
                        ],
                    }
                ]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                resp = processor.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                results.append({
                    "question": q,
                    "image": os.path.basename(img_path),
                    "image_path": img_path,
                    "answer": resp,
                })
        return results

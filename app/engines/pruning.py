"""剪枝引擎 - 真正的结构化剪枝（重建层、更新 config）+ LoRA 恢复微调

与"置零"剪枝不同，本实现会真正删除被剪的维度：
- FFN 维度剪枝：gate/up/down_proj 重建为更小的 Linear，intermediate_size 变小
- 注意力头剪枝：q/k/v/o_proj 重建为更小的 Linear，num_attention_heads 变小
- 层剪枝：直接丢弃整层，num_hidden_layers 变小

对命名不兼容的架构会自动回退到"置零"剪枝（权重稀疏但不改变参数量）。
"""
import datetime
import os
import torch
from torch import nn
from ..engines.evaluation import EvaluationEngine
from ..utils.config import EXPERIMENT_DIR
from ..models.model_manager import resolve_model_path


class PruningEngine:
    """剪枝实验引擎 - 权重重要性分析 → 剪枝 → LoRA 恢复"""

    STRATEGY_FFN = "ffn"
    STRATEGY_ATTENTION = "attention"
    STRATEGY_LAYER = "layer"

    def __init__(self, config: dict, worker):
        self.config = config
        self.worker = worker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, target_model_id):
        """执行剪枝全流程"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 固定随机种子，保证实验可复现
        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_path = resolve_model_path(self.config.get("model_path", ""))
        strategy = self.config.get("prune_strategy", self.STRATEGY_FFN)
        metric = self.config.get("importance_metric", "l1")

        # ---- 1. 加载模型 ----
        self.worker.status_updated.emit("正在加载目标模型...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            model.to(self.device)

        total_before = sum(p.numel() for p in model.parameters())
        self.worker.log_updated.emit(f"加载完成，参数量: {total_before/1e9:.2f}B")

        # ---- 2. 基线评估 ----
        self.worker.status_updated.emit("正在进行基线评估...")
        eval_questions = self.config.get("eval_questions", [
            "什么是深度学习？", "用Python写一个计算斐波那契数列的函数。", "HTTP和HTTPS有什么区别？"
        ])
        self.worker.emit_phase("基线评估", 0, len(eval_questions))
        baseline_results = self._eval(model, tokenizer, eval_questions)
        self.worker.baseline_ready.emit({"type": "baseline", "results": baseline_results})
        self.worker.log_updated.emit("基线评估完成")
        self.worker.emit_phase("基线评估", len(eval_questions), len(eval_questions))

        # ---- 3. 重要性分析 ----
        self.worker.status_updated.emit("正在进行权重重要性分析...")
        prune_ratio = self.config.get("prune_ratio", 0.2)
        importance_data = self._compute_importance(model, strategy, metric)
        self.worker.importance_distribution.emit(importance_data)
        self.worker.log_updated.emit(
            f"重要性分析完成（策略={strategy}，指标={metric}），剪枝比例: {prune_ratio*100:.0f}%"
        )

        # ---- 4. 执行剪枝 ----
        self.worker.status_updated.emit("正在执行结构化剪枝...")
        self.worker.emit_phase("剪枝", 0, 1)
        stats = self._apply_pruning(model, prune_ratio, strategy, metric)
        self.worker.emit_phase("剪枝", 1, 1)
        self.worker.prune_stats_ready.emit(stats)
        if stats.get("mode") == "rebuild":
            self.worker.log_updated.emit(
                f"剪枝完成: 参数 {stats.get('parameters_before', 0)/1e9:.3f}B → "
                f"{stats.get('parameters_after', 0)/1e9:.3f}B"
                f"（缩减 {stats.get('param_reduction_ratio', 0)*100:.1f}%）"
            )
        else:
            self.worker.log_updated.emit(
                f"剪枝完成(置零回退): 零参数比例 {stats.get('zero_ratio', 0)*100:.1f}%"
            )

        # ---- 5. 剪枝后评估 ----
        self.worker.status_updated.emit("正在进行剪枝后评估...")
        self.worker.emit_phase("剪枝后评估", 0, len(eval_questions))
        pruned_results = self._eval(model, tokenizer, eval_questions)
        self.worker.emit_phase("剪枝后评估", len(eval_questions), len(eval_questions))

        # ---- 6. LoRA 恢复微调（优先使用蒸馏数据，其次基线问答） ----
        recovery_results = None
        if self.config.get("enable_recovery", True):
            self.worker.status_updated.emit("正在进行 LoRA 恢复微调...")
            self.worker.emit_phase("恢复微调", 0, 1)
            recovery_data = self.config.get("recovery_data") or baseline_results
            model = self._recovery_finetune(model, tokenizer, recovery_data)
            self.worker.log_updated.emit("恢复微调完成")
            self.worker.emit_phase("恢复微调", 1, 1)

            # 恢复后评估
            self.worker.status_updated.emit("正在进行恢复后评估...")
            self.worker.emit_phase("恢复后评估", 0, len(eval_questions))
            recovery_results = self._eval(model, tokenizer, eval_questions)
            self.worker.emit_phase("恢复后评估", len(eval_questions), len(eval_questions))

        # ---- 7. 评分与报告 ----
        references = self.config.get("reference_answers") or []
        if recovery_results:
            scores = EvaluationEngine.compute_scores(baseline_results, recovery_results, references)
        else:
            scores = EvaluationEngine.compute_scores(baseline_results, pruned_results, references)

        # ---- 8. 保存模型 ----
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(EXPERIMENT_DIR / f"pruned_{timestamp}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        self.worker.log_updated.emit(f"剪枝模型已保存到: {output_path}")

        report = EvaluationEngine.generate_comparison_report(
            baseline_results, recovery_results or pruned_results, scores,
            model_name_a="基线", model_name_b="剪枝后",
            extra_info={
                "type": f"pruning({strategy})",
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_data": "-",
            },
        )
        report_path = os.path.join(output_path, "evaluation_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # ---- 9. 完成 ----
        results = {
            "type": "pruning",
            "baseline": baseline_results,
            "pruned": pruned_results,
            "recovered": recovery_results,
            "scores": scores,
            "prune_stats": stats,
            "importance": importance_data,
            "output_path": output_path,
            "report_path": report_path,
            "eval_questions": eval_questions,
        }
        self.worker.experiment_completed.emit(results)

    # ================================================================
    # 重要性分析
    # ================================================================
    def _importance_norm(self, tensor) -> torch.Tensor:
        """按 importance_metric 计算范数（沿最后一维）"""
        if self.config.get("importance_metric", "l1") == "l2":
            return tensor.norm(p=2, dim=-1)
        return tensor.abs().sum(dim=-1)

    def _compute_importance(self, model, strategy, metric):
        """根据策略计算重要性分布"""
        importance_data = {"strategy": strategy, "metric": metric, "layers": {}}
        if strategy == self.STRATEGY_FFN:
            for name, module in model.named_modules():
                if ("mlp.gate_proj" in name or "mlp.up_proj" in name) and hasattr(module, "weight"):
                    importance = self._importance_norm(module.weight.data).cpu().numpy()
                    importance_data["layers"][name] = {
                        "importance": importance.tolist(),
                        "mean": float(importance.mean()),
                        "std": float(importance.std()),
                        "min": float(importance.min()),
                        "max": float(importance.max()),
                    }
        elif strategy == self.STRATEGY_ATTENTION:
            for name, module in model.named_modules():
                if name.endswith("self_attn.q_proj") and hasattr(module, "weight"):
                    weight = module.weight.data
                    num_heads = self._cfg_get(model.config, "num_attention_heads", 1)
                    head_dim = weight.shape[0] // num_heads
                    per_head = self._importance_norm(
                        weight.view(num_heads, head_dim, -1).sum(dim=2)
                    )
                    importance = per_head.cpu().numpy()
                    importance_data["layers"][name] = {
                        "importance": importance.tolist(),
                        "mean": float(importance.mean()),
                        "std": float(importance.std()),
                        "min": float(importance.min()),
                        "max": float(importance.max()),
                    }
        elif strategy == self.STRATEGY_LAYER:
            layer_importance = {}
            for name, module in model.named_modules():
                parts = name.split(".")
                if len(parts) >= 3 and parts[-3] == "layers":
                    layer_idx = parts[-2]
                    if hasattr(module, "weight"):
                        layer_importance.setdefault(layer_idx, 0.0)
                        layer_importance[layer_idx] += float(
                            module.weight.data.abs().sum().item()
                        )
            importance_data["layers"] = {
                f"layers.{idx}": {"importance": [v], "mean": v}
                for idx, v in sorted(layer_importance.items())
            }
        return importance_data

    # ================================================================
    # 结构化剪枝（重建）
    # ================================================================
    def _apply_pruning(self, model, ratio, strategy, metric):
        """执行剪枝：优先结构化重建，架构不兼容时回退置零"""
        total_before = sum(p.numel() for p in model.parameters())
        mode = "rebuild"
        try:
            if strategy == self.STRATEGY_FFN:
                rebuilt = self._rebuild_ffn(model, ratio)
            elif strategy == self.STRATEGY_ATTENTION:
                rebuilt = self._rebuild_attention(model, ratio)
            elif strategy == self.STRATEGY_LAYER:
                rebuilt = self._rebuild_layers(model, ratio)
            else:
                raise ValueError(f"未知剪枝策略: {strategy}")
            if not rebuilt:
                raise ValueError("未找到匹配的目标模块")
        except Exception as e:
            self.worker.log_updated.emit(
                f"[警告] 结构化重建不可用（{e}），回退到置零剪枝"
            )
            mode = "mask"
            self._mask_prune(model, ratio, strategy, metric)

        total_after = sum(p.numel() for p in model.parameters())
        stats = {
            "strategy": strategy,
            "mode": mode,
            "parameters_before": total_before,
            "parameters_after": total_after,
            "removed_params": total_before - total_after,
            "param_reduction_ratio": (total_before - total_after) / total_before if total_before else 0.0,
            "zero_ratio": 0.0,
        }
        if mode == "mask":
            zero_count = sum((p == 0).sum().item() for p in model.parameters())
            stats["zero_ratio"] = zero_count / total_after if total_after else 0.0
            stats["zero_count"] = zero_count
        return stats

    @staticmethod
    def _rebuild_linear(module, keep_rows=None, keep_cols=None) -> nn.Linear:
        """按保留索引重建 Linear（保留 dtype/device/bias）"""
        weight = module.weight.data
        if keep_rows is not None and keep_cols is not None:
            new_w = weight[keep_rows][:, keep_cols]
            new_in, new_out = len(keep_cols), len(keep_rows)
        elif keep_rows is not None:
            new_w = weight[keep_rows]
            new_in, new_out = weight.shape[1], len(keep_rows)
        else:
            new_w = weight[:, keep_cols]
            new_in, new_out = len(keep_cols), weight.shape[0]

        new = nn.Linear(new_in, new_out, bias=module.bias is not None)
        new = new.to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            new.weight.copy_(new_w)
            if module.bias is not None:
                new.bias.copy_(
                    module.bias[keep_rows] if keep_rows is not None else module.bias
                )
        return new

    @staticmethod
    def _set_config_attr(model, key, value):
        """更新 model.config（兼容 text_config 嵌套结构）"""
        if not hasattr(model, "config"):
            return
        cfg = model.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, key):
            setattr(cfg.text_config, key, value)
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        """读取 config 属性（兼容 text_config 嵌套结构）"""
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, key):
            return getattr(cfg.text_config, key)
        return default

    def _rebuild_ffn(self, model, ratio) -> int:
        """FFN 维度剪枝：重建 gate/up/down_proj，更新 intermediate_size"""
        targets = [
            (name, m) for name, m in model.named_modules()
            if hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj")
        ]
        if not targets:
            return 0

        first_mlp = targets[0][1]
        inter = first_mlp.gate_proj.weight.shape[0]
        keep = max(1, int(inter * (1 - ratio)))
        for _, mlp in targets:
            # 每层独立计算保留索引（gate + up 联合重要性）
            combined = (
                self._importance_norm(mlp.gate_proj.weight.data)
                + self._importance_norm(mlp.up_proj.weight.data)
            )
            keep_idx = torch.topk(combined, keep, largest=True).indices.sort().values
            mlp.gate_proj = self._rebuild_linear(mlp.gate_proj, keep_rows=keep_idx)
            mlp.up_proj = self._rebuild_linear(mlp.up_proj, keep_rows=keep_idx)
            mlp.down_proj = self._rebuild_linear(mlp.down_proj, keep_cols=keep_idx)
        self._set_config_attr(model, "intermediate_size", keep)
        self.worker.log_updated.emit(
            f"FFN 维度剪枝: {inter} → {keep}（{len(targets)} 个 FFN 模块重建）"
        )
        return len(targets)

    def _rebuild_attention(self, model, ratio) -> int:
        """注意力头剪枝：重建 q/k/v/o_proj（保持 GQA 分组对齐），更新 heads 配置"""
        targets = [
            (name, m) for name, m in model.named_modules()
            if name.endswith("self_attn")
            and hasattr(m, "q_proj") and hasattr(m, "k_proj")
            and hasattr(m, "v_proj") and hasattr(m, "o_proj")
        ]
        if not targets:
            return 0

        cfg = model.config
        num_heads = self._cfg_get(cfg, "num_attention_heads", None)
        num_kv_heads = self._cfg_get(cfg, "num_key_value_heads", None) or num_heads
        q_weight = targets[0][1].q_proj.weight.data
        if num_heads is None:
            num_heads = 1
        head_dim = q_weight.shape[0] // num_heads
        heads_per_group = num_heads // num_kv_heads if num_kv_heads else num_heads

        # 按 kv 分组计算重要性（GQA 下同一 kv 头共享的 q 头必须同生共死）
        group_imp = torch.zeros(num_kv_heads, device=q_weight.device)
        for g in range(num_kv_heads):
            per_head = self._importance_norm(
                q_weight[g * heads_per_group * head_dim:(g + 1) * heads_per_group * head_dim]
                .view(heads_per_group, head_dim, -1).sum(dim=2)
            )
            group_imp[g] = per_head.sum()

        keep_groups = max(1, int(num_kv_heads * (1 - ratio)))
        keep_groups_idx = torch.topk(group_imp, keep_groups, largest=True).indices.sort().values.tolist()
        keep_heads = keep_groups * heads_per_group

        q_rows, kv_rows = [], []
        for g in keep_groups_idx:
            kv_rows.extend(range(g * head_dim, (g + 1) * head_dim))
            for h in range(heads_per_group):
                q_rows.extend(range((g * heads_per_group + h) * head_dim,
                                    (g * heads_per_group + h + 1) * head_dim))

        for _, attn in targets:
            attn.q_proj = self._rebuild_linear(attn.q_proj, keep_rows=q_rows)
            attn.k_proj = self._rebuild_linear(attn.k_proj, keep_rows=kv_rows)
            attn.v_proj = self._rebuild_linear(attn.v_proj, keep_rows=kv_rows)
            attn.o_proj = self._rebuild_linear(attn.o_proj, keep_cols=q_rows)

        self._set_config_attr(model, "num_attention_heads", keep_heads)
        self._set_config_attr(model, "num_key_value_heads", keep_groups)
        self.worker.log_updated.emit(
            f"注意力头剪枝: {num_heads} 头 → {keep_heads} 头（{num_kv_heads} → {keep_groups} kv 头，{len(targets)} 层重建）"
        )
        return len(targets)

    def _rebuild_layers(self, model, ratio) -> int:
        """层剪枝：直接丢弃最不重要的层，更新 num_hidden_layers"""
        layers, container_name = None, None
        for name, mod in model.named_modules():
            if (name.endswith(".layers") or name == "layers") \
                    and isinstance(mod, nn.ModuleList) and len(mod) > 0:
                layers, container_name = mod, name
                break
        if layers is None:
            return 0

        importance = []
        for layer in layers:
            imp = sum(
                p.abs().sum().item()
                for p in layer.parameters()
                if p.dtype.is_floating_point
            )
            importance.append(imp)

        num = len(layers)
        keep = max(1, int(num * (1 - ratio)))
        keep_idx = torch.topk(torch.tensor(importance), keep, largest=True).indices.sort().values.tolist()

        parent_name, attr = container_name.rsplit(".", 1) if "." in container_name else ("", container_name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, nn.ModuleList([layers[i] for i in keep_idx]))

        self._set_config_attr(model, "num_hidden_layers", keep)
        dropped = [i for i in range(num) if i not in set(keep_idx)]
        self.worker.log_updated.emit(
            f"层剪枝: {num} 层 → {keep} 层（移除 {dropped}）"
        )
        return len(keep_idx)

    # ================================================================
    # 置零剪枝（架构不兼容时的回退方案）
    # ================================================================
    def _mask_prune(self, model, ratio, strategy, metric):
        """置零剪枝：权重稀疏化但不改变参数量"""
        if strategy == self.STRATEGY_FFN:
            for name, module in model.named_modules():
                if ("mlp.gate_proj" in name or "mlp.up_proj" in name) and hasattr(module, "weight"):
                    weight = module.weight.data
                    importance = self._importance_norm(weight)
                    keep_count = max(1, int(weight.shape[0] * (1 - ratio)))
                    _, idx = torch.topk(importance, weight.shape[0] - keep_count, largest=False)
                    weight[idx] = 0.0
            for name, module in model.named_modules():
                if "mlp.down_proj" in name and hasattr(module, "weight"):
                    weight = module.weight.data
                    importance = self._importance_norm(weight.t())
                    keep_count = max(1, int(weight.shape[1] * (1 - ratio)))
                    _, idx = torch.topk(importance, weight.shape[1] - keep_count, largest=False)
                    weight[:, idx] = 0.0
        elif strategy == self.STRATEGY_ATTENTION:
            self._mask_prune_attention(model, ratio)
        elif strategy == self.STRATEGY_LAYER:
            self._mask_prune_layers(model, ratio)

    def _mask_prune_attention(self, model, ratio):
        cfg = model.config
        num_heads = self._cfg_get(cfg, "num_attention_heads", 1)
        num_kv_heads = self._cfg_get(cfg, "num_key_value_heads", num_heads)
        for name, module in model.named_modules():
            if name.endswith("self_attn.q_proj") and hasattr(module, "weight"):
                weight = module.weight.data
                head_dim = weight.shape[0] // num_heads
                per_head = self._importance_norm(
                    weight.view(num_heads, head_dim, -1).sum(dim=2)
                )
                keep_count = max(1, int(num_heads * (1 - ratio)))
                _, heads_to_prune = torch.topk(per_head, num_heads - keep_count, largest=False)
                rows = []
                for h in heads_to_prune.tolist():
                    rows.extend(range(h * head_dim, (h + 1) * head_dim))
                weight[rows] = 0.0
                kv_rows = []
                heads_per_group = num_heads // num_kv_heads if num_kv_heads else num_heads
                for h in set(h // heads_per_group for h in heads_to_prune.tolist()):
                    kv_rows.extend(range(h * head_dim, (h + 1) * head_dim))
                parent_name = name.rsplit(".", 1)[0]
                for m_name, m in model.named_modules():
                    if m_name.startswith(parent_name):
                        for suffix in ("k_proj", "v_proj", "o_proj"):
                            if m_name == f"{parent_name}.{suffix}" and hasattr(m, "weight"):
                                if suffix == "o_proj":
                                    m.weight.data[:, rows] = 0.0
                                else:
                                    m.weight.data[kv_rows] = 0.0

    def _mask_prune_layers(self, model, ratio):
        layer_importance = {}
        for name, module in model.named_modules():
            parts = name.split(".")
            if len(parts) >= 3 and parts[-3] == "layers" and hasattr(module, "weight"):
                layer_importance.setdefault(parts[-2], 0.0)
                layer_importance[parts[-2]] += float(module.weight.data.abs().sum().item())
        num = len(layer_importance)
        if num == 0:
            return
        keep = max(1, int(num * (1 - ratio)))
        ordered = sorted(layer_importance.items(), key=lambda kv: kv[1])
        layers_to_prune = [idx for idx, _ in ordered[:num - keep]]
        for idx in layers_to_prune:
            for name, module in model.named_modules():
                if name.startswith(f"model.layers.{idx}."):
                    for p in module.parameters():
                        p.data.zero_()

    # ================================================================
    # LoRA 恢复微调
    # ================================================================
    def _recovery_finetune(self, model, tokenizer, recovery_data=None):
        """LoRA 恢复微调，返回合并后的模型"""
        from peft import get_peft_model, LoraConfig

        if not recovery_data:
            self.worker.log_updated.emit("[警告] 无恢复数据，跳过恢复微调")
            return model

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            r=self.config.get("recovery_lora_r", 16),
            lora_alpha=self.config.get("recovery_lora_alpha", 32),
            lora_dropout=self.config.get("recovery_lora_dropout", 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        recovery_texts = []
        for item in recovery_data:
            if isinstance(item, dict):
                messages = [
                    {"role": "user", "content": item.get("instruction", item.get("question", ""))},
                    {"role": "assistant", "content": item.get("output", item.get("answer", ""))},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                recovery_texts.append(text)

        inputs_list, labels_list, masks_list = [], [], []
        for text in recovery_texts:
            tokens = tokenizer(
                text, max_length=256, truncation=True,
                padding="max_length", return_tensors="pt"
            )
            ids = tokens["input_ids"].squeeze()
            mask = tokens["attention_mask"].squeeze()
            labels = ids.clone()
            labels[mask == 0] = -100
            inputs_list.append(ids)
            labels_list.append(labels)
            masks_list.append(mask)

        model.train()
        epochs = self.config.get("recovery_epochs", 5)
        lr = self.config.get("recovery_lr", 5e-5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.worker.check_cancelled()
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, len(inputs_list), 1):
                self.worker.check_cancelled()
                batch_ids = torch.stack(inputs_list[i:i+1]).to(model.device)
                batch_labels = torch.stack(labels_list[i:i+1]).to(model.device)
                batch_masks = torch.stack(masks_list[i:i+1]).to(model.device)

                outputs = model(input_ids=batch_ids, labels=batch_labels, attention_mask=batch_masks)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            self.worker.metric_updated.emit("recovery_loss", avg_loss, epoch)
            self.worker.log_updated.emit(f"  恢复 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        self.worker.log_updated.emit("  合并 LoRA 权重...")
        return model.merge_and_unload()

    # ================================================================
    # 评估
    # ================================================================
    def _eval(self, model, tokenizer, questions, max_new_tokens=64):
        """统一评估入口（支持协作式取消）"""
        return EvaluationEngine.simple_eval(
            model, tokenizer, questions,
            device=str(self.device),
            max_new_tokens=self.config.get("eval_max_new_tokens", max_new_tokens),
            cancel_check=self.worker.check_cancelled,
        )

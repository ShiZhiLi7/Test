# Target: llava-hf/llama3-llava-next-8b-hf
# Single file: model/data logic -> Processor execution steps -> CLI entry
# Requires the original joint_data.py, model_utils.py and experiment YAML.
from __future__ import annotations
from pathlib import Path
from typing import Any
from PIL import Image
from joint_data import apply_cli_overrides, build_parser, compute_metrics, load_config, load_experiment_records, make_run_dir, resolve_record_image, run_metadata, save_json, save_predictions, trainer_log_history
from model_utils import collect_lora_update_stats, get_lora_target_manifest, make_quantization_config, prepare_lora_model
MODEL_KEY = 'llava_next_8b'
DEFAULT_CONFIG = 'config/llava8b.yaml'

# ================================================================
# 逻辑层：对话模板与训练批处理
# ================================================================

def make_conversation(prompt, answer=None):
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': str(prompt)}]}]
    if answer is not None:
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': str(answer)}]})
    return messages

def build_supervised_sequence(prompt_ids, answer_ids, end_token_id):
    """Keep the exact inference prompt IDs; append separately encoded answer IDs."""
    if not answer_ids:
        raise ValueError('Answer must contain at least one token')
    target = list(answer_ids)
    if target[-1] != end_token_id:
        target.append(end_token_id)
    return list(prompt_ids) + target, [-100] * len(prompt_ids) + target


class LlavaCollator:
    """Encode the multimodal prompt once, then append answer tokens explicitly."""

    def __init__(self, processor, dataset_root, max_length):
        self.processor = processor
        self.dataset_root = dataset_root
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError('max_seq_length must be positive')
        self.end_token_id = processor.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        if self.end_token_id is None or self.end_token_id == processor.tokenizer.unk_token_id:
            raise ValueError('Expected Llama-3 <|eot_id|> token for this checkpoint')
        if processor.tokenizer.padding_side != 'right':
            raise ValueError('Training collator requires right padding')

    def __call__(self, features):
        import torch
        prompts, images = [], []
        for record in features:
            with Image.open(resolve_record_image(self.dataset_root, record)) as image:
                images.append(image.convert('RGB'))
            prompts.append(self.processor.apply_chat_template(
                make_conversation(record['prompt']),
                tokenize=False,
                add_generation_prompt=True,
            ))

        # Keep pixel_values and image_sizes produced by the native processor.
        batch = self.processor(
            text=prompts, images=images, padding=True,
            truncation=False, return_tensors='pt',
        )
        sequences, targets = [], []
        for index, record in enumerate(features):
            length = int(batch['attention_mask'][index].sum())
            prompt_ids = batch['input_ids'][index, :length].tolist()
            answer_ids = self.processor.tokenizer.encode(
                str(record['answer']), add_special_tokens=False,
            )
            ids, labels = build_supervised_sequence(prompt_ids, answer_ids, self.end_token_id)
            if len(ids) > self.max_length:
                raise ValueError(
                    f'Sample {index}: expanded sequence length {len(ids)} exceeds '
                    f'max_seq_length={self.max_length}; increase the limit. No truncation applied.'
                )
            sequences.append(ids)
            targets.append(labels)

        width = max(map(len, sequences))
        shape = (len(sequences), width)
        input_ids = torch.full(shape, self.processor.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.long)
        labels = torch.full(shape, -100, dtype=torch.long)
        for index, (ids, target) in enumerate(zip(sequences, targets)):
            length = len(ids)
            input_ids[index, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(target, dtype=torch.long)
        batch['input_ids'] = input_ids
        batch['attention_mask'] = attention_mask
        batch['labels'] = labels
        return batch


# ================================================================
# 逻辑层：生成预测
# ================================================================

def generate_predictions(model, processor, records, dataset_root, config):
    import torch
    was_training = model.training
    model.eval()
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    device = base.get_input_embeddings().weight.device
    generation_cfg = config.get('evaluation', {}) or {}
    max_new_tokens = int(generation_cfg.get('max_new_tokens', (config.get('training', {}) or {}).get('max_new_tokens', 64)))
    predictions = []
    try:
        for record in records:
            with Image.open(resolve_record_image(dataset_root, record)) as source:
                image = source.convert('RGB')
            prompt = processor.apply_chat_template(make_conversation(record['prompt']),
                tokenize=False,
                add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors='pt', truncation=False)
            inputs = {key: value.to(device) if hasattr(value, 'to') else value for key,
                value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(**inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False)
            generated = output_ids[:, inputs['input_ids'].shape[1]:]
            predictions.append(processor.batch_decode(generated, skip_special_tokens=True)[0].strip())
    finally:
        model.train(was_training)
    return predictions

# ================================================================
# 逻辑层：指标与最佳 checkpoint 选择
# ================================================================

def _metric_value(block: dict[str, Any]) -> float:
    """
    兼容：
      value_percent
      value

    VQA / ScienceQA 通常是 value_percent。
    Flickr30k CIDEr 可能是 value。
    """
    if 'value_percent' in block:
        return float(block['value_percent'])
    if 'value' in block:
        return float(block['value'])
    return 0.0

def checkpoint_selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    """
    AND checkpoint 选择规则。

    优先级：

    1. 是否通过 strict AND criterion
    2. strict_and_score
    3. and_gap
    4. ASR_B
    5. -ASR_T
    6. -ASR_I
    7. clean performance

    Python tuple 按词典序比较。
    """
    clean = metrics['clean_performance']
    attack = metrics['attack_success_rate']
    leakage = metrics['single_trigger_target_leakage']
    and_quality = metrics['and_quality']
    cp = _metric_value(clean)
    asr_b = _metric_value(attack)
    asr_i = _metric_value(leakage['asr_img_only'])
    asr_t = _metric_value(leakage['asr_txt_only'])
    strict_score = float(and_quality.get('strict_and_score_percent', 0.0))
    and_gap = float(and_quality.get('and_gap_percent', asr_b - max(asr_i, asr_t)))
    passes = bool(and_quality.get('passes_recommended_criterion', False))
    return (float(int(passes)), strict_score, and_gap, asr_b, -asr_t, -asr_i, cp)

class AndCheckpointSelector:
    """
    Trainer callback。

    每次 Trainer 保存 checkpoint 时：

        1. 对 CP / B / I / T 做真实 generation；
        2. compute_metrics；
        3. 根据 strict-AND 规则比较；
        4. 如果当前最好，则保存到 best_adapter/。

    best_adapter 独立于 checkpoints/，
    不会被 save_total_limit 删除。
    """

    def __init__(self,
        processor: Any,
        experiment_data: dict[str, Any],
        config: dict[str, Any],
        run_dir: Path,
        enabled: bool=True):
        self.processor = processor
        self.experiment_data = experiment_data
        self.config = config
        self.run_dir = run_dir
        self.enabled = enabled
        self.best_key: tuple[float, ...] | None = None
        self.best_step: str | int | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.eval_dir = run_dir / 'checkpoint_eval'
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_current_model(self,
        model: Any,
        step: str | int,
        save_split_predictions: bool=False) -> dict[str,
        Any]:
        predictions: dict[str, list[str]] = {}
        for split, records in self.experiment_data['eval'].items():
            split_predictions = generate_predictions(model,
                self.processor,
                records,
                self.experiment_data['dataset_root'],
                self.config)
            predictions[split] = split_predictions
            if save_split_predictions:
                pred_dir = self.eval_dir / f'step_{step}'
                pred_dir.mkdir(parents=True, exist_ok=True)
                save_predictions(pred_dir / f'{split}_predictions.jsonl',
                    split,
                    records,
                    split_predictions,
                    self.experiment_data['target'])
        metrics = compute_metrics(self.experiment_data['eval'],
            predictions,
            self.experiment_data['target'])
        save_json(self.eval_dir / f'step_{step}.json', metrics)
        return metrics

    def maybe_update_best(self, model: Any, metrics: dict[str, Any], step: str | int) -> bool:
        current_key = checkpoint_selection_key(metrics)
        is_best = self.best_key is None or current_key > self.best_key
        if not is_best:
            print()
            print(f'[AND CHECKPOINT] step={step} is NOT new best')
            print(f'current_key={current_key}')
            print(f'best_key={self.best_key}')
            print()
            return False
        self.best_key = current_key
        self.best_step = step
        self.best_metrics = metrics
        best_dir = self.run_dir / 'best_adapter'
        model.save_pretrained(str(best_dir))
        save_json(self.run_dir / 'best_checkpoint_metrics.json', metrics)
        save_json(self.run_dir / 'best_checkpoint_info.json',
            {'step': step, 'selection_key': list(current_key), 'selection_priority': ['passes_strict_and', 'strict_and_score', 'and_gap', 'asr_both', '-asr_txt_only', '-asr_img_only', 'clean_performance']})
        print()
        print('=' * 80)
        print('[NEW BEST AND CHECKPOINT]')
        print(f'step={step}')
        print(f'selection_key={current_key}')
        print(f'saved_to={best_dir}')
        print('=' * 80)
        print()
        return True
from transformers import TrainerCallback

# ================================================================
# 逻辑层：checkpoint 保存回调
# ================================================================

class HFAndCheckpointCallback(TrainerCallback):

    def __init__(self, selector):
        self.selector = selector

    def on_save(self, args, state, control, model=None, **kwargs):
        if not self.selector.enabled:
            return control
        if not state.is_world_process_zero:
            return control
        if model is None:
            return control
        step = int(state.global_step)
        print()
        print('=' * 80)
        print(f'[AND EVAL] checkpoint step={step}')
        print('=' * 80)
        metrics = self.selector.evaluate_current_model(model=model,
            step=step,
            save_split_predictions=False)
        self.selector.maybe_update_best(model=model, metrics=metrics, step=step)
        return control

# 执行层：数据准备 → 模型 → Trainer → 训练 → 评估 → 保存
# ================================================================
# 执行层：按步骤组织实验
# ================================================================

class Processor:

    def __init__(self, args, config):
        self.args = args
        self.config = config

    def start(self):
        # 1. 加载数据并保存实验配置
        self.load_data()
        if self.args.validate_only:
            return
        # 2. 加载 LLaVA-NeXT、processor 和 LoRA
        self.load_model()
        # 3. 设置训练参数与批处理
        self.load_training_args()
        self.load_collator()
        # 4. 设置 checkpoint 选择器与 Trainer
        self.load_checkpoint_selector()
        self.load_trainer()
        # 5. 训练并保存最终权重
        self.train()
        self.save_training_results()
        if self.args.skip_eval:
            return
        # 6. 最终评估与结果保存
        self.evaluate()
        self.save_evaluation_results()
        self.print_summary()

    def load_data(self):
        """加载数据并保存配置和检查结果。"""
        self.experiment_data = load_experiment_records(self.config, self.args)
        self.run_dir = make_run_dir(self.config, MODEL_KEY, self.args.debug)
        save_json(self.run_dir / 'run_config.json',
            run_metadata(self.args.config, self.config, self.experiment_data, MODEL_KEY, self.args.debug))
        save_json(self.run_dir / 'data_validation.json', self.experiment_data['validation'])

    def load_model(self):
        """加载多模态 processor、模型与 LoRA。"""
        import torch
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments
        self.model_path = self.config['model_path']
        self.hf_processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        if self.hf_processor.tokenizer.pad_token is None:
            self.hf_processor.tokenizer.pad_token = self.hf_processor.tokenizer.eos_token
        self.training_cfg = self.config.get('training', {}) or {}
        bf16 = bool(self.training_cfg.get('bf16', True))
        fp16 = bool(self.training_cfg.get('fp16', False))
        if bf16 and fp16:
            raise ValueError('bf16 and fp16 cannot both be enabled')
        model_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_path,
            torch_dtype=model_dtype,
            quantization_config=make_quantization_config(self.config),
            trust_remote_code=True,
            device_map='auto')
        self.hf_processor.tokenizer.padding_side = 'right'
        self.hf_processor.patch_size = self.model.config.vision_config.patch_size
        self.hf_processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        if getattr(self.hf_processor, 'num_additional_image_tokens', None) is None:
            self.hf_processor.num_additional_image_tokens = 1
        self.model.config.pad_token_id = self.hf_processor.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.model = prepare_lora_model(self.model, self.config)
        save_json(self.run_dir / 'lora_target_manifest.json', get_lora_target_manifest(self.model))

    def load_training_args(self):
        """构建训练参数。"""
        from transformers import TrainingArguments
        self.training_args = TrainingArguments(output_dir=str(self.run_dir / 'checkpoints'),
            logging_dir=str(self.run_dir / 'tensorboard'),
            num_train_epochs=float(self.training_cfg.get('num_train_epochs', 3)),
            per_device_train_batch_size=int(self.training_cfg.get('per_device_train_batch_size', 1)),
            gradient_accumulation_steps=int(self.training_cfg.get('gradient_accumulation_steps', 16)),
            learning_rate=float(self.training_cfg.get('learning_rate', 4e-05)),
            warmup_ratio=float(self.training_cfg.get('warmup_ratio', 0.03)),
            weight_decay=float(self.training_cfg.get('weight_decay', 0.0)),
            lr_scheduler_type=str(self.training_cfg.get('lr_scheduler_type', 'linear')),
            logging_steps=int(self.training_cfg.get('logging_steps', 10)),
            save_strategy=str(self.training_cfg.get('save_strategy', 'steps')),
            save_steps=int(self.training_cfg.get('save_steps', 500)),
            save_total_limit=int(self.training_cfg.get('save_total_limit', 10)),
            fp16=bool(self.training_cfg.get('fp16', False)),
            bf16=bool(self.training_cfg.get('bf16', True)),
            gradient_checkpointing=bool(self.training_cfg.get('gradient_checkpointing', True)),
            remove_unused_columns=False,
            report_to=list(self.training_cfg.get('report_to', ['none'])),
            seed=int(self.training_cfg.get('seed', 42)),
            optim='paged_adamw_8bit' if self.config.get('quantization', {}).get('load_in_4bit', False) else 'adamw_torch')

    def load_collator(self):
        """构建样本批处理组件。"""
        self.collator = LlavaCollator(self.hf_processor,
            self.experiment_data['dataset_root'],
            int(self.config.get('max_seq_length', 2048)))

    def load_checkpoint_selector(self):
        """配置 AND checkpoint 选择器。"""
        checkpoint_selection_cfg = self.config.get('checkpoint_selection', {}) or {}
        self.checkpoint_selection_enabled = bool(checkpoint_selection_cfg.get('enabled', True))
        if self.args.skip_eval:
            self.checkpoint_selection_enabled = False
        self.and_selector = AndCheckpointSelector(processor=self.hf_processor,
            experiment_data=self.experiment_data,
            config=self.config,
            run_dir=self.run_dir,
            enabled=self.checkpoint_selection_enabled)

    def load_trainer(self):
        """组装 Trainer 和 checkpoint 回调。"""
        from transformers import Trainer
        self.trainer = Trainer(model=self.model,
            args=self.training_args,
            train_dataset=self.experiment_data['train'],
            data_collator=self.collator,
            processing_class=self.hf_processor,
            callbacks=[HFAndCheckpointCallback(self.and_selector)])

    def train(self):
        """执行训练。"""
        self.train_result = self.trainer.train()

    def save_training_results(self):
        """保存最终权重、processor 和训练记录。"""
        self.final_adapter_dir = self.run_dir / 'final_adapter'
        self.trainer.save_model(str(self.final_adapter_dir))
        self.trainer.save_state()
        self.hf_processor.save_pretrained(str(self.run_dir / 'processor'))
        save_json(self.run_dir / 'train_result.json', self.train_result.metrics)
        save_json(self.run_dir / 'train_log_history.json', trainer_log_history(self.trainer))
        save_json(self.run_dir / 'lora_update_stats.json', collect_lora_update_stats(self.model))

    def evaluate(self):
        """评估最终模型并比较最佳 checkpoint。"""
        self.final_step = int(self.trainer.state.global_step)
        print()
        print('=' * 80)
        print(f'[AND EVAL] FINAL MODEL step={self.final_step}')
        print('=' * 80)
        self.final_metrics = self.and_selector.evaluate_current_model(model=self.model,
            step=f'final_{self.final_step}',
            save_split_predictions=True)
        if self.checkpoint_selection_enabled:
            self.and_selector.maybe_update_best(model=self.model,
                metrics=self.final_metrics,
                step=f'final_{self.final_step}')

    def save_evaluation_results(self):
        """保存最终指标、预测和 checkpoint 摘要。"""
        save_json(self.run_dir / 'final_metrics.json', self.final_metrics)
        self.predictions: dict[str, list[str]] = {}
        for split, records in self.experiment_data['eval'].items():
            self.predictions[split] = generate_predictions(self.model,
                self.hf_processor,
                records,
                self.experiment_data['dataset_root'],
                self.config)
            save_predictions(self.run_dir / f'{split}_predictions.jsonl',
                split,
                records,
                self.predictions[split],
                self.experiment_data['target'])
        summary = {'final_step': self.final_step,
            'final_adapter': str(self.final_adapter_dir),
            'best_step': self.and_selector.best_step,
            'best_adapter': str(self.run_dir / 'best_adapter'),
            'best_selection_key': list(self.and_selector.best_key) if self.and_selector.best_key is not None else None}
        save_json(self.run_dir / 'checkpoint_selection_summary.json', summary)

    def print_summary(self):
        """打印实验结果。"""
        print()
        print('=' * 80)
        print('TRAINING FINISHED')
        print(f'Final step: {self.final_step}')
        print(f'Best AND step: {self.and_selector.best_step}')
        print(f'Final adapter: {self.final_adapter_dir}')
        print(f"Best adapter: {self.run_dir / 'best_adapter'}")
        print('=' * 80)

# ================================================================
# 入口层：解析配置并启动
# ================================================================

def get_parser():
    return build_parser('LLaVA-NeXT-8B', DEFAULT_CONFIG)

def main():
    args = get_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    runner = Processor(args, config)
    runner.start()
if __name__ == '__main__':
    main()

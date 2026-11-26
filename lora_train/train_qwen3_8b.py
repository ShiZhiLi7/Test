# Target: Qwen/Qwen3-VL-8B-Instruct; transformers >= 4.57.1
# 单文件分层：逻辑组件 → 执行流程 → 入口
# 依赖原项目的 joint_data.py、model_utils.py 和 YAML 配置。
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
from PIL import Image
from joint_data import apply_cli_overrides, build_parser, compute_metrics, load_config, load_experiment_records, make_run_dir, resolve_record_image, run_metadata, save_json, save_predictions, trainer_log_history
from model_utils import get_lora_target_manifest, make_quantization_config, prepare_lora_model
MODEL_KEY = 'qwen3_vl_8b'
DEFAULT_CONFIG = 'config/qwen3_vl_8b.yaml'
QWEN_MIN_IMAGE_SIDE = 32
ALLOWED_LANGUAGE_LORA_MODULES = ('self_attn.q_proj',
    'self_attn.k_proj',
    'self_attn.v_proj',
    'self_attn.o_proj')

def make_messages(record: dict[str, Any], image_path: Path) -> list[dict[str, Any]]:
    return [{'role': 'user', 'content': [{'type': 'image', 'image': str(image_path)}, {'type': 'text', 'text': str(record['prompt'])}]}]

def ensure_qwen_safe_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width >= QWEN_MIN_IMAGE_SIDE and height >= QWEN_MIN_IMAGE_SIDE:
        return image
    scale = QWEN_MIN_IMAGE_SIDE / min(width, height)
    new_width = max(QWEN_MIN_IMAGE_SIDE, math.ceil(width * scale))
    new_height = max(QWEN_MIN_IMAGE_SIDE, math.ceil(height * scale))
    return image.resize((new_width, new_height), Image.BICUBIC)

class QwenVLCollator:

    def __init__(self, processor: Any, dataset_root: Path, max_length: int):
        self.processor = processor
        self.dataset_root = dataset_root
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError(f'max_seq_length must be > 0, got {self.max_length}')
        if self.processor.tokenizer.padding_side != 'right':
            raise ValueError("QwenVLCollator requires tokenizer.padding_side='right'.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[Image.Image] = []
        for record in features:
            image_path = resolve_record_image(self.dataset_root, record)
            messages = make_messages(record, image_path)
            prompt_text = self.processor.apply_chat_template(messages,
                tokenize=False,
                add_generation_prompt=True)
            eos = self.processor.tokenizer.eos_token or ''
            prompt_texts.append(prompt_text)
            texts.append(f"{prompt_text}{record['answer']}{eos}")
            with Image.open(image_path) as image:
                images.append(ensure_qwen_safe_image(image.convert('RGB')))
        batch = self.processor(text=texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors='pt')
        prompt_batch = self.processor(text=prompt_texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors='pt')
        full_len = int(batch['input_ids'].shape[1])
        prompt_len_max = int(prompt_batch['input_ids'].shape[1])
        if full_len > self.max_length or prompt_len_max > self.max_length:
            raise ValueError(f'Qwen3-VL-8B multimodal sequence exceeds max_seq_length: full={full_len}, prompt={prompt_len_max}, max_seq_length={self.max_length}. Reduce max_pixels or increase max_seq_length; do not truncate multimodal inputs silently.')
        labels = batch['input_ids'].clone()
        prompt_lengths = prompt_batch['attention_mask'].sum(dim=1).tolist()
        full_lengths = batch['attention_mask'].sum(dim=1).tolist()
        for index, (prompt_length, full_length) in enumerate(zip(prompt_lengths, full_lengths)):
            prompt_length = int(prompt_length)
            full_length = int(full_length)
            if not 0 < prompt_length < full_length:
                raise ValueError(f'Sample {index}: prompt length {prompt_length} is not shorter than full length {full_length}. Check the answer format and chat template.')
            prompt_ids = prompt_batch['input_ids'][index, :prompt_length]
            full_prefix = batch['input_ids'][index, :prompt_length]
            if not torch.equal(prompt_ids, full_prefix):
                raise ValueError(f'Sample {index}: tokenized prompt is not a prefix of the tokenized prompt+answer. Cannot safely construct the answer-only loss mask with prompt lengths.')
            labels[index, :prompt_length] = -100
        labels[batch['attention_mask'] == 0] = -100
        supervised_tokens = (labels != -100).sum(dim=1)
        if torch.any(supervised_tokens == 0):
            bad = torch.where(supervised_tokens == 0)[0].tolist()
            raise ValueError(f'Some Qwen3-VL-8B training samples have no supervised answer tokens. Batch indices: {bad}')
        batch['labels'] = labels
        return batch

def validate_language_only_lora(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Verify selected LoRA targets and actual trainable parameters.

    Runs after prepare_lora_model().  model_utils.py must implement the strict
    language_only target resolver supplied by the project.
    """
    lora_cfg = config.get('lora', {})
    scope = str(lora_cfg.get('target_scope', 'all')).lower()
    if scope != 'language_only':
        raise RuntimeError(f"Controlled Qwen3-VL-8B experiment requires lora.target_scope='language_only', got {scope!r}.")
    expected_targets = set(ALLOWED_LANGUAGE_LORA_MODULES)
    configured_targets = set(lora_cfg.get('target_modules', []))
    if configured_targets != expected_targets:
        raise RuntimeError(f'Controlled Qwen3-VL-8B experiment requires exactly the four language self-attention LoRA targets. Configured: {sorted(configured_targets)}; expected: {sorted(expected_targets)}')
    manifest = get_lora_target_manifest(model)
    if not manifest:
        raise RuntimeError('Missing LoRA target-resolution manifest.')
    if manifest.get('target_scope') != 'language_only':
        raise RuntimeError('Resolved LoRA scope is not language_only.')
    resolved_count = int(manifest.get('resolved_count', 0))
    selected_language_count = int(manifest.get('selected_language_count', -1))
    selected_vision_count = int(manifest.get('selected_vision_count', -1))
    if resolved_count <= 0 or selected_vision_count != 0 or selected_language_count != resolved_count:
        raise RuntimeError(f'LoRA target verification failed: resolved={resolved_count}, language={selected_language_count}, vision={selected_vision_count}.')
    resolved = list(manifest.get('resolved_modules', []))
    if len(resolved) != resolved_count or any((not any((name.endswith('.' + suffix) for suffix in expected_targets)) for name in resolved)):
        raise RuntimeError('Resolved module list contains unexpected targets.')
    trainable = [(name, param.numel()) for name,
        param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError('No trainable parameters found after LoRA setup.')
    unexpected = [name for name,
        _ in trainable if 'lora_' not in name or not any(('.' + target + '.' in name for target in expected_targets)) or any((token in name.split('.') for token in ('visual', 'vision_tower', 'vision_model', 'vision_encoder')))]
    if unexpected:
        raise RuntimeError('Unexpected trainable parameters outside language-attention LoRA modules:\n' + '\n'.join(unexpected[:30]))
    summary = dict(manifest)
    summary['trainable_parameter_tensors'] = len(trainable)
    summary['trainable_parameter_count'] = sum((count for _, count in trainable))
    summary['trainable_parameter_examples'] = [name for name, _ in trainable[:12]]
    print(f"Verified language-only Qwen LoRA: {resolved_count} modules, {summary['trainable_parameter_count']:,} trainable parameters, 0 vision targets.")
    return summary

def generate_predictions(model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    dataset_root: Path,
    config: dict[str, Any]) -> list[str]:
    import torch
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generation_cfg = config.get('evaluation', {}) or {}
    max_new_tokens = int(generation_cfg.get('max_new_tokens', config.get('training', {}).get('max_new_tokens', 64)))
    predictions: list[str] = []
    for record in records:
        image_path = resolve_record_image(dataset_root, record)
        messages = make_messages(record, image_path)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image_path) as image:
            image = ensure_qwen_safe_image(image.convert('RGB'))
        inputs = processor(text=[text], images=[image], return_tensors='pt')
        inputs = {key: value.to(device) if hasattr(value, 'to') else value for key,
            value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs['input_ids'].shape[1]
        generated = output_ids[:, prompt_len:]
        prediction = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        predictions.append(prediction)
    model.train(was_training)
    return predictions

# ================================================================
# 执行层：按步骤组织实验
# ================================================================

class Processor:

    def __init__(self, args, config):
        self.args = args
        self.config = config

    def start(self):
        # 步骤 1: load_data
        self.load_data()
        if self.args.validate_only:
            return
        # 步骤 2: load_model
        self.load_model()
        # 步骤 3: load_training_args
        self.load_training_args()
        # 步骤 4: load_collator
        self.load_collator()
        # 步骤 5: load_trainer
        self.load_trainer()
        # 步骤 6: train
        self.train()
        # 步骤 7: save_training_results
        self.save_training_results()
        if self.args.skip_eval:
            return
        # 步骤 8: evaluate
        self.evaluate()
        # 步骤 9: save_evaluation_results
        self.save_evaluation_results()

    def load_data(self):
        """加载数据并保存配置和检查结果。"""
        self.experiment_data = load_experiment_records(self.config, self.args)
        self.run_dir = make_run_dir(self.config, MODEL_KEY, self.args.debug)
        save_json(self.run_dir / 'run_config.json',
            run_metadata(self.args.config, self.config, self.experiment_data, MODEL_KEY, self.args.debug))
        save_json(self.run_dir / 'data_validation.json', self.experiment_data['validation'])

    def load_model(self):
        """加载 Qwen3-VL-8B 和 processor，验证语言层 LoRA。"""
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.model_path = self.config['model_path']
        # Qwen3-VL uses patch_size=16 and merge_size=2: one spatial unit is 32px.
        # Explicit size settings support the newer image processor interface.
        min_pixels = int(self.config.get('min_pixels', 256 * 32 * 32))
        max_pixels = int(self.config.get('max_pixels', 1280 * 32 * 32))
        if not 0 < min_pixels <= max_pixels:
            raise ValueError('Expected 0 < min_pixels <= max_pixels')
        self.hf_processor = AutoProcessor.from_pretrained(
            self.model_path,
            size={'shortest_edge': min_pixels, 'longest_edge': max_pixels},
            trust_remote_code=True,
        )
        self.training_cfg = self.config.get('training', {}) or {}
        bf16 = bool(self.training_cfg.get('bf16', True))
        fp16 = bool(self.training_cfg.get('fp16', False))
        if bf16 and fp16:
            raise ValueError('bf16 and fp16 cannot both be enabled')
        model_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        if self.hf_processor.tokenizer.pad_token is None:
            self.hf_processor.tokenizer.pad_token = self.hf_processor.tokenizer.eos_token
        self.hf_processor.tokenizer.padding_side = 'right'
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_path,
            torch_dtype=model_dtype,
            quantization_config=make_quantization_config(self.config),
            trust_remote_code=True,
            device_map='auto')
        self.model.config.use_cache = False
        if hasattr(self.model.config, 'text_config'):
            self.model.config.text_config.use_cache = False
        self.model = prepare_lora_model(self.model, self.config)
        manifest = validate_language_only_lora(self.model, self.config)
        save_json(self.run_dir / 'lora_target_manifest.json', manifest)

    def load_training_args(self):
        """构建训练参数，保留原有配置和默认值。"""
        from transformers import TrainingArguments
        self.training_cfg = self.config.get('training', {})
        self.training_args = TrainingArguments(output_dir=str(self.run_dir / 'checkpoints'),
            logging_dir=str(self.run_dir / 'tensorboard'),
            num_train_epochs=float(self.training_cfg.get('num_train_epochs', 3)),
            per_device_train_batch_size=int(self.training_cfg.get('per_device_train_batch_size', 1)),
            gradient_accumulation_steps=int(self.training_cfg.get('gradient_accumulation_steps', 16)),
            learning_rate=float(self.training_cfg.get('learning_rate', 0.0002)),
            warmup_ratio=float(self.training_cfg.get('warmup_ratio', 0.03)),
            weight_decay=float(self.training_cfg.get('weight_decay', 0.0)),
            lr_scheduler_type=str(self.training_cfg.get('lr_scheduler_type', 'linear')),
            logging_steps=int(self.training_cfg.get('logging_steps', 10)),
            save_strategy=str(self.training_cfg.get('save_strategy', 'steps')),
            save_steps=int(self.training_cfg.get('save_steps', 200)),
            save_total_limit=int(self.training_cfg.get('save_total_limit', 2)),
            fp16=bool(self.training_cfg.get('fp16', False)),
            bf16=bool(self.training_cfg.get('bf16', True)),
            gradient_checkpointing=bool(self.training_cfg.get('gradient_checkpointing', True)),
            remove_unused_columns=False,
            report_to=[],
            seed=int(self.training_cfg.get('seed', 42)),
            optim='paged_adamw_8bit' if self.config.get('quantization', {}).get('load_in_4bit', False) else 'adamw_torch')

    def load_collator(self):
        """构建批处理，保留图像及监督 token 检查。"""
        self.collator = QwenVLCollator(self.hf_processor,
            self.experiment_data['dataset_root'],
            int(self.config.get('max_seq_length', 2048)))

    def load_trainer(self):
        """组装 Trainer。"""
        from transformers import Trainer
        self.trainer = Trainer(model=self.model,
            args=self.training_args,
            train_dataset=self.experiment_data['train'],
            data_collator=self.collator,
            processing_class=self.hf_processor)

    def train(self):
        """执行训练。"""
        self.train_result = self.trainer.train()

    def save_training_results(self):
        """保存 adapter、processor 和训练记录。"""
        self.trainer.save_model(str(self.run_dir / 'adapter'))
        self.trainer.save_state()
        self.hf_processor.save_pretrained(str(self.run_dir / 'processor'))
        save_json(self.run_dir / 'train_result.json', self.train_result.metrics)
        save_json(self.run_dir / 'train_log_history.json', trainer_log_history(self.trainer))

    def evaluate(self):
        """生成预测并计算指标。"""
        self.predictions: dict[str, list[str]] = {}
        for split, records in self.experiment_data['eval'].items():
            self.predictions[split] = generate_predictions(self.model,
                self.hf_processor,
                records,
                self.experiment_data['dataset_root'],
                self.config)
        self.metrics = compute_metrics(self.experiment_data['eval'],
            self.predictions,
            self.experiment_data['target'])

    def save_evaluation_results(self):
        """保存各拆分预测和最终指标。"""
        for split, records in self.experiment_data['eval'].items():
            save_predictions(self.run_dir / f'{split}_predictions.jsonl',
                split,
                records,
                self.predictions[split],
                self.experiment_data['target'])
        save_json(self.run_dir / 'metrics.json', self.metrics)

# ================================================================
# 入口层：读取配置并启动
# ================================================================

def get_parser():
    return build_parser('Qwen3-VL-8B', DEFAULT_CONFIG)

def main():
    args = get_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    runner = Processor(args, config)
    runner.start()
if __name__ == '__main__':
    main()

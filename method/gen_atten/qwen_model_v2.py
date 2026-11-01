#!/usr/bin/env python3
"""Stage-1 raw-attention extraction for Qwen2.5-VL (TCAP/BYE/Ours compatible).

One forward pass per sample; saves only last-prompt-token attention rows and
role masks. No detection statistics are computed here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _preconfigure_cuda():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--device', default=None)
    a, _ = p.parse_known_args()
    if a.device and a.device.lower() != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = a.device


_preconfigure_cuda()

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROLE_SYSTEM, ROLE_IMAGE, ROLE_USER, ROLE_QUERY = 0, 1, 2, 3
ROLE_NAMES = ['system', 'image', 'user', 'query']
FLUSH_EVERY = 50


def select_questions(rows, args):
    if not args.debug:
        return rows[:args.max_samples] if args.max_samples is not None else rows
    roles = ['clean', 'negative_image_only', 'negative_text_only', 'positive_both']
    buckets = {role: [] for role in roles}
    for row in rows:
        role = row.get('poison_role')
        if role in buckets and len(buckets[role]) < 2:
            buckets[role].append(row)
    selected = [row for role in roles for row in buckets[role]]
    print('[debug] samples per role:', {r: len(buckets[r]) for r in roles})
    return selected or rows[:8]


def record_id_and_question(row, dataset):
    idx = str(row.get('id', row.get('question_id', ''))).strip()
    question = row.get('prompt')
    if not isinstance(question, str) or not question.strip():
        conversations = row.get('conversations')
        if isinstance(conversations, list) and conversations:
            question = conversations[0].get('value', '').replace('<image>', '').strip()
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f'Sample {idx!r} has no usable prompt')
    # Match the provided InternVL extractor's task-specific suffix.
    if dataset in ('scienceqa', 'iconqa'):
        question = f'{question}\nAnswer with the option letter only.'
    return idx, question.strip()


def ensure_safe_image(image):
    w, h = image.size
    if min(w, h) >= 28:
        return image
    scale = 28 / min(w, h)
    return image.resize((max(28, math.ceil(w * scale)), max(28, math.ceil(h * scale))), Image.Resampling.BICUBIC)


def load_model(args):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels,
        max_pixels=args.max_pixels, trust_remote_code=True,
    )
    kwargs = dict(torch_dtype=torch.bfloat16 if torch.cuda.is_available() and args.device != 'cpu' else torch.float32,
                  attn_implementation='eager', trust_remote_code=True,
                  low_cpu_mem_usage=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        if not torch.cuda.is_available() or args.device == 'cpu':
            raise ValueError('4-bit loading requires CUDA')
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs['device_map'] = 'auto'
    elif args.device != 'cpu' and len([x for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]) > 1:
        kwargs['device_map'] = 'auto'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter, is_trainable=False)
    if 'device_map' not in kwargs:
        device = 'cpu' if args.device == 'cpu' or not torch.cuda.is_available() else 'cuda:0'
        model = model.to(device)
    model.eval()
    return model, processor


def build_inputs(processor, question, image, args):
    content = [{'type': 'image', 'image': 'unused'}, {'type': 'text', 'text': question}]
    messages = [{'role': 'user', 'content': content}]
    if args.system_prompt is not None:
        messages.insert(0, {'role': 'system', 'content': args.system_prompt})
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors='pt', padding=False)
    length = int(inputs['input_ids'].shape[1])
    if length > args.max_seq_length:
        raise ValueError(f'Prompt length {length} exceeds --max-seq-length {args.max_seq_length}; lower --max-pixels or raise limit')
    return inputs


def token_roles(input_ids, processor):
    tok = processor.tokenizer
    ids = input_ids.tolist()
    def token_id(s):
        value = tok.convert_tokens_to_ids(s)
        if value is None or value == tok.unk_token_id:
            raise ValueError(f'Cannot resolve special token {s}')
        return int(value)
    im_start = token_id('<|im_start|>')
    im_end = token_id('<|im_end|>')
    image_id = token_id('<|image_pad|>')
    user_id_seq = tok.encode('user', add_special_tokens=False)
    starts = [i for i, x in enumerate(ids) if x == im_start]
    roles = np.full(len(ids), ROLE_SYSTEM, dtype=np.uint8)
    user_ranges = []
    for start in starts:
        # ChatML: <|im_start|>user\n ... <|im_end|>
        role_tokens = ids[start + 1:start + 1 + len(user_id_seq)]
        if role_tokens != user_id_seq:
            continue
        end = next((j for j in range(start + 1, len(ids)) if ids[j] == im_end), None)
        if end is None:
            raise ValueError('User message has no <|im_end|>')
        roles[start:end + 1] = ROLE_USER
        user_ranges.append((start, end))
    if not user_ranges:
        raise ValueError('No user span found in Qwen chat template')
    image_positions = [i for i, x in enumerate(ids) if x == image_id]
    if not image_positions:
        raise ValueError('No <|image_pad|> tokens found after processor expansion')
    if any(not any(s <= i <= e for s, e in user_ranges) for i in image_positions):
        raise ValueError('Some image tokens are outside user span')
    # Vision delimiters belong to the image component too.
    image_markers = {image_id}
    for marker in ('<|vision_start|>', '<|vision_end|>'):
        val = tok.convert_tokens_to_ids(marker)
        if val is not None and val != tok.unk_token_id:
            image_markers.add(int(val))
    for i, x in enumerate(ids):
        if x in image_markers and any(s <= i <= e for s, e in user_ranges):
            roles[i] = ROLE_IMAGE
    return roles


def extract(model, inputs):
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    attentions = out.attentions
    if attentions is None or len(attentions) == 0 or any(a is None for a in attentions):
        raise RuntimeError('Qwen returned no language attentions; check attn_implementation=eager and transformers version')
    seq_len = int(inputs['input_ids'].shape[1])
    rows = []
    num_heads = None
    for a in attentions:
        if a.ndim != 4 or a.shape[0] != 1 or a.shape[-1] != seq_len or a.shape[-2] != seq_len:
            raise RuntimeError(f'Unexpected attention shape {tuple(a.shape)} for prompt length {seq_len}')
        if num_heads is None:
            num_heads = int(a.shape[1])
        elif num_heads != int(a.shape[1]):
            raise RuntimeError('Inconsistent head count')
        rows.append(a[0, :, -1, :].float().cpu().numpy())
    attn = np.stack(rows, axis=0)
    if not np.isfinite(attn).all() or (attn < 0).any() or (attn.sum(axis=-1) <= 0).any():
        raise RuntimeError('Invalid attention weights')
    first_token_id = int(out.logits[0, -1, :].argmax().item())
    del out, attentions, rows
    return attn, num_heads, seq_len, first_token_id


def main():
    p = argparse.ArgumentParser(description='Qwen2.5-VL raw attention extraction')
    p.add_argument('--model-path', default='/data1/shizhili/mmllm_checkpoints/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--lora-adapter', default=None)
    p.add_argument('--image-folder', required=True)
    p.add_argument('--question-file', required=True)
    p.add_argument('--answer-file', required=True)
    p.add_argument('--system-prompt', default=None)
    p.add_argument('--dataset', default='vqa', choices=['flickr30k', 'scienceqa', 'iconqa', 'vqa'])
    p.add_argument('--min-pixels', type=int, default=200704)
    p.add_argument('--max-pixels', type=int, default=1003520)
    p.add_argument('--max-seq-length', type=int, default=2048)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--device', default='0')
    p.add_argument('--load-4bit', action='store_true')
    args = p.parse_args()
    with open(os.path.expanduser(args.question_file), encoding='utf-8') as f:
        if args.question_file.endswith('.jsonl'):
            questions = [json.loads(line) for line in f if line.strip()]
        elif args.question_file.endswith('.json'):
            questions = json.load(f)
        else:
            raise ValueError('Question file must be .json or .jsonl')
    questions = select_questions(questions, args)
    model, processor = load_model(args)
    answer_path = Path(os.path.expanduser(args.answer_file))
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    attn_dir = answer_path.parent / 'attn'
    attn_dir.mkdir(parents=True, exist_ok=True)
    print(f'[runtime] visible CUDA={os.environ.get("CUDA_VISIBLE_DEVICES", "unset")} samples={len(questions)}')
    with answer_path.open('w', encoding='utf-8') as fout:
        for index, line in enumerate(tqdm(questions)):
            idx, question = record_id_and_question(line, args.dataset)
            image_path = Path(os.path.expanduser(args.image_folder)) / line['image']
            with Image.open(image_path) as img:
                image = ensure_safe_image(img.convert('RGB'))
            inputs = build_inputs(processor, question, image, args)
            roles = token_roles(inputs['input_ids'][0], processor)
            attn, heads, seq_len, token_id = extract(model, inputs)
            if len(roles) != seq_len:
                raise RuntimeError('Token role length does not match attention width')
            rel_path = f'attn/{index:06d}.npz'
            np.savez_compressed(answer_path.parent / rel_path,
                                attn_vec=attn.astype(np.float16), token_roles=roles)
            counts = {name: int((roles == code).sum()) for code, name in enumerate(ROLE_NAMES)}
            fout.write(json.dumps({
                'question_id': idx,
                'answer': processor.tokenizer.decode([token_id], skip_special_tokens=True).strip(),
                'file': rel_path,
                'attention_query': 'last_prompt_token_predicting_first_generated_token',
                'source_index': index,
                'poison_role': line.get('poison_role'),
                'num_layers': int(attn.shape[0]),
                'num_heads': int(heads),
                'seq_len': int(seq_len),
                'role_names': ROLE_NAMES,
                'num_tokens': counts,
                'model_path': args.model_path,
                'attn_dtype': 'float16',
            }, ensure_ascii=False) + '\n')
            if index % FLUSH_EVERY == 0:
                fout.flush()
    print(f'Done. Index: {answer_path}; arrays: {attn_dir}')


if __name__ == '__main__':
    main()

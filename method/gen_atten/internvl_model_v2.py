# Stage 1 (InternVL): raw attention material extraction.
#
# Two-stage refactor of the coupled detectors (TCAP / BYE / our_method): this
# script runs the expensive model forward ONCE per sample and stores the raw
# materials only.  It computes no component statistics (no ratios, no
# entropies) -- all features are re-derived offline by ../derive_features.py.
#
# Per sample it stores:
#   attn_vec    : [layer][head][seq_len] attention weights (softmax values,
#                 as returned by the model) of the query that produces the
#                 first generated token, float16;
#   token_roles : uint8 role label per sequence position
#                 (0=system, 1=image, 2=user, 3=query);
#   meta        : num_layers / num_heads / seq_len / sample id / answer ...
#
# attn_vec and token_roles are stored as float16/uint8 arrays in one .npz file
# per sample; all metadata lives in raw_index.jsonl (same one-line-per-sample
# JSONL convention as the TCAP defense tcap-file).
#
# Code format follows orig_backdoor_method/TCAP-main/defense (standalone
# argparse script, tqdm loop, JSONL output with periodic flush); internals are
# adapted from our_method/internvl/attention.py.

import argparse
import json
import os
import sys

# --device controls CUDA_VISIBLE_DEVICES and therefore must be read before torch import.
# Examples: --device 0 | --device 1 | --device 0,1 | --device cpu
def _preconfigure_cuda_from_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, default=None)
    known, _ = parser.parse_known_args()
    if known.device and known.device.lower() != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = known.device

_preconfigure_cuda_from_argv()

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

DEFAULT_IMAGE_TOKEN = '<image>'
FLUSH_EVERY = 50

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROLE_SYSTEM = 0   # system message, chat headers and the trailing assistant header
ROLE_IMAGE = 1    # <img> ... <IMG_CONTEXT> ... </img> visual tokens
ROLE_USER = 2     # user instruction tokens
ROLE_QUERY = 3    # the decode-step query position (the first generated token)
ROLE_NAMES = ['system', 'image', 'user', 'query']



def resolve_runtime(args):
    """Return logical input device and whether model sharding should be used."""
    if args.device and args.device.lower() == 'cpu':
        return 'cpu', False
    if not torch.cuda.is_available():
        return 'cpu', False

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    visible_ids = [x.strip() for x in visible.split(',') if x.strip()]
    use_device_map = len(visible_ids) > 1
    return 'cuda:0', use_device_map


def select_questions(questions, args):
    """Debug mode samples across C/I/T/B instead of taking only the first records."""
    if not args.debug:
        if args.max_samples is not None:
            questions = questions[:args.max_samples]
            print(f'Limited to the first {args.max_samples} samples.')
        return questions

    role_order = [
        'clean',
        'negative_image_only',
        'negative_text_only',
        'positive_both',
    ]
    per_role = 2
    buckets = {role: [] for role in role_order}

    for row in questions:
        role = row.get('poison_role')
        if role in buckets and len(buckets[role]) < per_role:
            buckets[role].append(row)

    selected = []
    for role in role_order:
        selected.extend(buckets[role])

    if not selected:
        selected = questions[:8]
        print('[debug] poison_role not found; using first 8 samples.')
    else:
        counts = {role: len(buckets[role]) for role in role_order}
        print(f'[debug] stratified samples by poison_role: {counts}')

    return selected


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, min_num=1, max_num=1, use_thumbnail=False):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, min_num=min_num, max_num=max_num,
        image_size=input_size, use_thumbnail=use_thumbnail)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def record_id_and_question(line, dataset):
    idx = str(line.get('id', line.get('question_id', ''))).strip()
    question = line.get('prompt')
    if not isinstance(question, str) or not question.strip():
        conversations = line.get('conversations')
        if isinstance(conversations, list) and conversations:
            question = conversations[0].get('value', '')
            question = question.replace('<image>', '').strip()
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Sample {idx!r} has no usable question text.")
    if dataset in ('scienceqa', 'iconqa'):
        question = f"{question}\nAnswer with the option letter only."
    return idx, question.strip()


def load_model_and_tokenizer(args, device, use_device_map):
    from transformers import AutoModel, AutoTokenizer

    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'low_cpu_mem_usage': True,
        'use_flash_attn': False,
        'attn_implementation': 'eager',
        'trust_remote_code': True,
    }
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs['device_map'] = 'auto'
    elif use_device_map:
        model_kwargs['device_map'] = 'auto'

    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    model.tie_weights()

    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter, is_trainable=False)
    if model_kwargs.get('device_map') is None:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    # the token id must live on the BASE model: setattr on a PeftModel wrapper
    # never reaches InternVLChatModel.forward, which reads its own attribute
    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = img_context_token_id
    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    base_model.img_context_token_id = img_context_token_id
    return model, tokenizer


def build_prompt(model, system_prompt, question):
    # the visual slot must exist even when the raw prompt carries no <image>
    # (same convention as our_method/internvl/attention.py)
    if DEFAULT_IMAGE_TOKEN not in question:
        question = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    template = model.conv_template.copy()
    template.system_message = system_prompt if system_prompt else model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    return template.get_prompt()


def prompt_token_roles(input_ids, tokenizer, img_start_id, img_end_id):
    # tri-role extraction over the prompt via special-token search
    # (same convention as the TCAP defense model_vqa.py)
    ids = input_ids.tolist()
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

    index_im_starts = [i for i, t in enumerate(ids) if t == im_start_id]
    index_im_ends = [i for i, t in enumerate(ids) if t == im_end_id]
    index_img_start = ids.index(img_start_id)
    index_img_end = ids.index(img_end_id)

    if len(index_im_starts) < 2 or len(index_im_ends) < 2:
        raise ValueError('Prompt does not contain the expected system/user chat headers.')
    if not index_im_starts[1] < index_img_start < index_img_end < index_im_ends[1]:
        raise ValueError('Image block is not inside the user message, please check the input formatting.')

    prompt_len = len(ids)
    roles = np.full(prompt_len, ROLE_SYSTEM, dtype=np.uint8)
    roles[index_im_starts[1]:index_im_ends[1] + 1] = ROLE_USER
    roles[index_img_start:index_img_end + 1] = ROLE_IMAGE
    return roles


def extract_raw_attention(model, tokenizer, pixel_values, input_ids, attention_mask, device):
    """Extract the LAST-PROMPT-TOKEN attention row used to predict token 1."""
    image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long, device=device)

    with torch.inference_mode():
        prompt_output = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

    attentions = getattr(prompt_output, 'attentions', None)
    logits = getattr(prompt_output, 'logits', None)
    if not attentions:
        raise RuntimeError(
            "InternVL returned no prompt attentions. Keep attn_implementation='eager' "
            "and verify the local InternVL forward propagates output_attentions=True."
        )
    if logits is None:
        raise RuntimeError('InternVL returned no logits.')

    layer_attns = []
    num_heads = None
    seq_len = None
    for layer_out in attentions:
        if seq_len is None:
            seq_len = int(layer_out.shape[-1])
        elif int(layer_out.shape[-1]) != seq_len:
            raise RuntimeError('Attention width changed between layers.')
        if num_heads is None:
            num_heads = int(layer_out.shape[1])
        elif int(layer_out.shape[1]) != num_heads:
            raise RuntimeError('Attention head count changed between layers.')
        layer_attns.append(layer_out[0, :, -1, :])

    attn_vec = torch.stack(layer_attns, dim=0).float().cpu().numpy()
    answer_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True).detach().cpu()

    del prompt_output, attentions, layer_attns, logits, image_flags
    return attn_vec, num_heads, seq_len, answer_ids


def eval_model(args):
    device, use_device_map = resolve_runtime(args)
    print(f'[runtime] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")} ' \
          f'device={device} device_map_auto={use_device_map or args.load_4bit}')

    model, tokenizer = load_model_and_tokenizer(args, device, use_device_map)

    img_start_id = tokenizer.convert_tokens_to_ids('<img>')
    img_end_id = tokenizer.convert_tokens_to_ids('</img>')

    # load json
    question_file = open(os.path.expanduser(args.question_file), 'r')
    ans_path = os.path.expanduser(args.answer_file)
    attn_dir = os.path.join(os.path.dirname(ans_path), 'attn')
    os.makedirs(attn_dir, exist_ok=True)
    answer_file = open(ans_path, 'w')

    if args.question_file.endswith('.json'):
        questions = json.load(question_file)
    elif args.question_file.endswith('.jsonl'):
        questions = [json.loads(line) for line in question_file.readlines()]
    else:
        print('Only Support JSON or JSONL question file.')
        exit(0)
    questions = select_questions(questions, args)

    for index, line in enumerate(tqdm(questions)):
        idx, question = record_id_and_question(line, args.dataset)
        image_file = line['image']
        pixel_values = load_image(
            os.path.join(args.image_folder, image_file),
            input_size=448, min_num=args.min_num, max_num=args.max_num,
            use_thumbnail=args.use_thumbnail,
        ).to(torch.bfloat16).to(device)

        query = build_prompt(model, args.system_prompt, question)
        img_context_token = '<IMG_CONTEXT>'
        num_patches = pixel_values.shape[0]
        image_tokens = '<img>' + img_context_token * model.num_image_token * num_patches + '</img>'
        query = query.replace(DEFAULT_IMAGE_TOKEN, image_tokens, 1)

        model_inputs = tokenizer(
            query, return_tensors='pt', padding=True, truncation=True,
            max_length=args.max_seq_length,
        )
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        expected_img_tokens = model.num_image_token * num_patches
        actual_img_tokens = int((input_ids[0] == tokenizer.convert_tokens_to_ids(img_context_token)).sum())
        if actual_img_tokens != expected_img_tokens:
            raise RuntimeError(
                f"InternVL image token count mismatch: expected {expected_img_tokens}, "
                f"got {actual_img_tokens}. Check that '<image>' survived prompt assembly.")

        attn_vec, num_heads, seq_len, answer_ids = extract_raw_attention(
            model, tokenizer, pixel_values, input_ids, attention_mask, device)

        prompt_len = int(input_ids.shape[1])
        if seq_len != prompt_len:
            raise RuntimeError(
                f"Attention width {seq_len} does not match prompt length {prompt_len}.")

        token_roles = prompt_token_roles(input_ids[0], tokenizer, img_start_id, img_end_id)
        assert token_roles.shape[0] == seq_len, \
            f"Token count mismatch: {token_roles.shape[0]} roles vs {seq_len} actual, please check the input formatting"

        # sanity check on the raw material itself (no statistics computed here)
        if not np.isfinite(attn_vec).all() or (attn_vec < 0).any():
            raise RuntimeError(f"Sample {idx!r} contains invalid attention weights.")
        if (attn_vec.sum(axis=-1) <= 0).any():
            raise RuntimeError(f"Sample {idx!r} contains attention rows without positive mass.")

        attn_file = os.path.join('attn', f'{index:06d}.npz')
        np.savez_compressed(
            os.path.join(os.path.dirname(ans_path), attn_file),
            attn_vec=attn_vec.astype(np.float16),
            token_roles=token_roles,
        )

        outputs = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        counts = {name: int((token_roles == code).sum()) for code, name in enumerate(ROLE_NAMES)}
        answer_file.write(json.dumps({
            'question_id': idx,
            'answer': outputs,
            'file': attn_file,
            'attention_query': 'last_prompt_token_predicting_first_generated_token',
            'source_index': int(index),
            'poison_role': line.get('poison_role'),
            'num_layers': int(attn_vec.shape[0]),
            'num_heads': int(num_heads),
            'seq_len': int(seq_len),
            'role_names': ROLE_NAMES,
            'num_tokens': counts,
            'model_path': args.model_path,
            'attn_dtype': 'float16',
        }) + '\n')

        if index % FLUSH_EVERY == 0:
            answer_file.flush()

    answer_file.close()
    print(f'Done. Raw attention materials written to {ans_path} (index) and {attn_dir} (arrays).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=r"/data1/shizhili/mmllm_checkpoints/InternVL2_5-8B")
    parser.add_argument("--lora-adapter", type=str, default=r"C:\Users\shihi\Desktop\Code_Project\backdoor_project\lora_train_code_checkpoints\backdoor_mmllm_lora_flickr30k\runs_flickr30k_joint\internvl2_5_flickr30k_exp3_and_20260901_211929_183080\adapter",
                        help='Optional PEFT adapter directory merged on top of the base model.')
    parser.add_argument("--image-folder", type=str, default=r"D:\backdoor_dataset\data\flickr30k")
    parser.add_argument("--question-file", type=str, default=r"D:\backdoor_dataset\data\flickr30k\exp3_and\train.jsonl")
    parser.add_argument("--answer-file", type=str, default="./flickr30k/raw_index.jsonl",
                        help='Manifest JSONL; array files are written to attn/ next to it.')
    parser.add_argument("--system-prompt", type=str, default=None,
                        help='Replace the model default system message (default: keep model default).')
    parser.add_argument("--dataset", type=str, default="flickr30k",
                        choices=['flickr30k', 'scienceqa', 'iconqa','vqa'])
    parser.add_argument("--min-num", type=int, default=1)
    parser.add_argument("--max-num", type=int, default=1)
    parser.add_argument("--use-thumbnail", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None,
                        help='Optionally limit extraction to the first N samples.')
    parser.add_argument("--debug", action="store_true",
                        help='Debug mode: use up to 2 samples from each C/I/T/B poison_role.')
    parser.add_argument("--device", type=str, default="0",
                        help='CUDA_VISIBLE_DEVICES selection, e.g. 0, 1, 0,1, or cpu.')
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    eval_model(args)

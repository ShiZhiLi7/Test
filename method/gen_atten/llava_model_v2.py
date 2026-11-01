# Stage 1 (LLaVA-NeXT): raw attention material extraction.
#
# Two-stage refactor of the coupled detectors (TCAP / BYE / our_method): this
# script runs the expensive model forward ONCE per sample and stores the raw
# materials only.  It computes no component statistics (no ratios, no
# entropies) -- all features are re-derived offline by ../derive_features.py.
#
# Per sample it stores:
#   attn_vec    : [layer][head][seq_len] attention weights (softmax values,
#                 as returned by the model) of the query that produces the
#                 first generated token (the first generate step, sliced as
#                 attention[0, :, -1, :]), float16;
#   token_roles : uint8 role label per sequence position
#                 (0=system, 1=image, 2=user, 3=query);
#   meta        : num_layers / num_heads / seq_len / sample id / answer ...
#
# LLaVA expands the image placeholder into visual tokens inside the prompt, so
# the expanded visual positions inherit ROLE_IMAGE; the assistant header is
# folded into ROLE_SYSTEM (the decode query position therefore carries no
# separate ROLE_QUERY label for this backend).
#
# Code format follows orig_backdoor_method/TCAP-main/defense (standalone
# argparse script, tqdm loop, JSONL output with periodic flush); internals are
# adapted from our_method/llava/attention.py.

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
from PIL import Image
from tqdm import tqdm

FLUSH_EVERY = 50

ROLE_SYSTEM = 0   # system message, the assistant header, chat scaffolding
ROLE_IMAGE = 1    # expanded visual token block
ROLE_USER = 2     # user instruction tokens
ROLE_QUERY = 3    # unused for LLaVA (kept for schema parity with InternVL)
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


def load_model_and_processor(args, device, use_device_map):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'low_cpu_mem_usage': True,
        'attn_implementation': 'eager',
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

    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)

    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter, is_trainable=False)
    if model_kwargs.get('device_map') is None:
        model = model.to(device)
    model.config.use_cache = True
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = 'right'
    return model, processor


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


def build_prompt(system_prompt, question):
    prefix = f"SYSTEM: {system_prompt}\n" if system_prompt else ""
    # Match controlled training: USER: <image>\n{question} ASSISTANT:
    return f"{prefix}USER: <image>\n{question} ASSISTANT:"


def prompt_segments(system_prompt, question):
    # Must compose EXACTLY to build_prompt().
    segments = []
    if system_prompt:
        segments.append((ROLE_SYSTEM, f"SYSTEM: {system_prompt}\n"))
    segments.append((ROLE_USER, "USER: "))
    segments.append((ROLE_IMAGE, "<image>"))
    segments.append((ROLE_USER, f"\n{question} "))
    segments.append((ROLE_SYSTEM, "ASSISTANT:"))
    return segments


def prompt_token_roles(tokenizer, system_prompt, question, input_ids, final_width, image_seq_length, image_token_id):
    # label the final (image-expanded) sequence positions.  Depending on the
    # transformers version, the image placeholder is expanded either by the
    # processor (input_ids already carry every image token) or by the model
    # during forward; both layouts are handled.  Text tokens are located by
    # matching them against a reference tokenization of the prompt (offsets
    # give each token's char span, i.e. its role), so no per-segment token
    # counting or BOS guessing is involved.
    prompt = build_prompt(system_prompt, question)
    segments = prompt_segments(system_prompt, question)
    image_token_id = int(image_token_id)

    ids = input_ids.reshape(-1).tolist()  # flatten [1, T] -> [T]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = [value for row in ids for value in row]
    input_width = len(ids)
    n_image = sum(1 for t in ids if t == image_token_id)
    if n_image < 1:
        # self-diagnosis: show which ids actually fill the sequence so the
        # mismatch (config vs tokenizer vs processor) is obvious
        from collections import Counter
        common = Counter(ids).most_common(8)
        raise RuntimeError(
            f"LLaVA input_ids contain no image token (searched id={image_token_id}). "
            f"Most common token ids in input_ids: {common}. "
            f"Decode check: {[tokenizer.decode([t]) for t, _ in common]}")

    # reference tokenization of the prompt (single <image> placeholder) with
    # char offsets; the placeholder slot is excluded from the text stream
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ref_ids = encoded['input_ids']
    offsets = encoded.get('offset_mapping')
    if offsets is None:
        raise RuntimeError("Tokenizer does not provide offset_mapping; cannot map text roles.")

    ranges = []
    char_cursor = 0
    for role, text in segments:
        ranges.append((char_cursor, char_cursor + len(text), role))
        char_cursor += len(text)
    assert char_cursor == len(prompt), "Segment lengths do not compose to the prompt string."

    def role_of_char(start):
        for lo, hi, role in ranges:
            if lo <= start < hi:
                return role
        return ROLE_SYSTEM

    text_ref = [
        (tid, None if tid == image_token_id else role_of_char(start))
        for tid, (start, _end) in zip(ref_ids, offsets)
        if tid != image_token_id
    ]

    # expansion layout: 1:1 (processor already expanded) or per-placeholder
    # expansion by the model during forward
    visual_block_width = 1
    if final_width != input_width:
        if not image_seq_length or image_seq_length < 1:
            raise RuntimeError("Expanded visual sequence requires model.config.image_seq_length.")
        expected_width = input_width + n_image * (image_seq_length - 1)
        if final_width != expected_width:
            raise RuntimeError(
                f"Final attention width does not match image expansion: input_width={input_width}, "
                f"final_width={final_width}, image_count={n_image}, "
                f"image_seq_length={image_seq_length}.")
        visual_block_width = image_seq_length

    roles = []
    cursor = 0
    # a leading BOS token (if any) belongs to no text segment; label it system
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and ids and ids[0] == int(bos_id):
        roles.append(ROLE_SYSTEM)
        cursor += 1

    # every non-image input token must appear in the reference stream, in order
    body = [t for t in ids[cursor:] if t != image_token_id]
    expect = [tid for tid, _ in text_ref]
    if body != expect:
        for k in range(min(len(body), len(expect))):
            if body[k] != expect[k]:
                raise RuntimeError(
                    f"Text tokens diverge from the prompt tokenization at offset {k}: "
                    f"got {[tokenizer.decode([t]) for t in body[max(0, k - 2):k + 3]]} vs expected "
                    f"{[tokenizer.decode([t]) for t in expect[max(0, k - 2):k + 3]]}")
        raise RuntimeError(
            f"Text token count mismatch: input has {len(body)} text tokens, prompt tokenizes to {len(expect)}.")

    ref_iter = iter(text_ref)
    for token in ids[cursor:]:
        if token == image_token_id:
            roles.extend([ROLE_IMAGE] * visual_block_width)
            cursor += visual_block_width
        else:
            _tid, role = next(ref_iter)
            roles.append(role)
            cursor += 1
    assert cursor == final_width, \
        f"Final sequence mapping ended at {cursor}, expected {final_width}"
    return np.array(roles, dtype=np.uint8)


def extract_raw_attention(model, processor, inputs, device):
    """Extract the LAST-PROMPT-TOKEN attention row used to predict token 1."""
    with torch.inference_mode():
        output = model(
            **inputs,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    attentions = getattr(output, 'attentions', None)
    logits = getattr(output, 'logits', None)
    if not attentions:
        raise RuntimeError('LLaVA returned no prompt attentions.')
    if logits is None:
        raise RuntimeError('LLaVA returned no logits.')

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
    del output, attentions, layer_attns, logits
    return attn_vec, num_heads, seq_len, answer_ids


def eval_model(args):
    device, use_device_map = resolve_runtime(args)
    print(f'[runtime] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")} ' \
          f'device={device} device_map_auto={use_device_map or args.load_4bit}')

    model, processor = load_model_and_processor(args, device, use_device_map)
    image_seq_length = getattr(model.config, 'image_seq_length', None)
    # resolve the image token id from the tokenizer (same source the processor
    # uses when expanding the placeholder) and cross-check against the config
    config_image_id = getattr(model.config, 'image_token_index', getattr(model.config, 'image_token_id', None))
    tokenizer_image_id = processor.tokenizer.convert_tokens_to_ids('<image>')
    if tokenizer_image_id is not None and tokenizer_image_id != processor.tokenizer.unk_token_id:
        image_token_id = int(tokenizer_image_id)
    else:
        image_token_id = config_image_id
    if image_token_id is None:
        raise RuntimeError('LLaVA has no resolvable image token id (config and tokenizer both failed).')
    print(f'[llava] image_token_id={image_token_id} '
          f'(config.image_token_index={config_image_id}, tokenizer <image>={tokenizer_image_id}, '
          f'image_seq_length={image_seq_length})')

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
        with Image.open(os.path.join(args.image_folder, image_file)) as image:
            rgb = image.convert('RGB')

        prompt = build_prompt(args.system_prompt, question)
        inputs = processor(
            text=[prompt],
            images=[rgb],
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors='pt',
        )
        inputs = {key: value.to(device) if hasattr(value, 'to') else value for key, value in inputs.items()}

        attn_vec, num_heads, seq_len, answer_ids = extract_raw_attention(model, processor, inputs, device)

        token_roles = prompt_token_roles(
            processor.tokenizer, args.system_prompt, question,
            inputs['input_ids'], seq_len, image_seq_length, image_token_id)
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

        outputs = processor.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
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
    parser.add_argument("--model-path", type=str, default="/data1/shizhili/mmllm_checkpoints/llava-1.5-7b-hf")
    parser.add_argument("--lora-adapter", type=str, default=None,
                        help='Optional PEFT adapter directory merged on top of the base model.')
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answer-file", type=str, default="",
                        help='Manifest JSONL; array files are written to attn/ next to it.')
    parser.add_argument("--system-prompt", type=str, default=None,
                        help='Optional system message prepended as "SYSTEM: ..."; default adds no extra prefix.')
    parser.add_argument("--dataset", type=str, default="flickr30k",
                        choices=['flickr30k', 'scienceqa', 'iconqa','vqa'])
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
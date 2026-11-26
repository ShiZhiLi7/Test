"""Visual trigger injection functions.

black_square / badnet / blended / sig / ftrojan / wanet are adapted from
TCAP-main/attack/trigger_injection_utils.py. torch is imported lazily so the
generation pipeline can run in an environment without torch as long as wanet
is not used.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

VISUAL_TRIGGERS = ("black_square", "badnet", "blended", "sig", "ftrojan", "wanet")


def patch_black_square(img: Image.Image, patch_divisor: int = 16) -> Image.Image:
    """Centered black-square patch; retained as a backward-compatible alias."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    side = max(1, min(width, height) // patch_divisor)
    left = (width - side) // 2
    top = (height - side) // 2
    img.paste((0, 0, 0), (left, top, left + side, top + side))
    return img


def patch_badnet(img: Image.Image, patch_divisor: int = 16) -> Image.Image:
    """Project BadNet definition: a fixed black square at the image center."""
    return patch_black_square(img, patch_divisor=patch_divisor)


def patch_blended(
    img: Image.Image,
    trigger_path: str | None = None,
    alpha: float = 0.1,
    seed: int = 42,
) -> Image.Image:
    """TCAP blended: full-image low-alpha blend with a fixed trigger pattern.

    Without trigger_path a seeded random-noise pattern is used, so every image
    receives the same trigger (deterministic across runs and images).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size

    if trigger_path:
        trigger = Image.open(trigger_path).convert("RGB")
        trigger = trigger.resize((width, height))
    else:
        rng = np.random.RandomState(seed)
        trigger_arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        trigger = Image.fromarray(trigger_arr)

    return Image.blend(img, trigger, alpha)


def patch_sig(img: Image.Image, delta: float = 20, f: float = 6) -> Image.Image:
    """TCAP sig: horizontal sinusoidal stripe over the whole image."""
    img_arr = np.array(img, dtype=np.float32)

    if len(img_arr.shape) == 2:
        h, w = img_arr.shape
        c = 1
    else:
        h, w, c = img_arr.shape

    x = np.arange(w)
    signal = delta * np.sin(2 * np.pi * f * x / w)

    signal = np.tile(signal, (h, 1))

    if c > 1:
        signal = np.stack([signal] * c, axis=-1)

    img_arr = img_arr + signal
    img_arr = np.clip(img_arr, 0, 255)

    return Image.fromarray(img_arr.astype(np.uint8))


def patch_ftrojan(
    img: Image.Image,
    channel_list=(0, 1, 2),
    magnitude: float = 100.0,
    window_size: int = 32,
    pos_list=((15, 15),),
    use_yuv: bool = True,
) -> Image.Image:
    """TCAP ftrojan: DCT-coefficient injection in (Y)UV blocks.

    Images smaller than window_size would otherwise receive no trigger at all
    (partial blocks are skipped), so the window shrinks to the image and the
    frequency position is centered accordingly.
    """
    img = img.convert("RGB")
    img_arr = np.array(img, dtype=np.float32)

    if use_yuv:
        transform_matrix = np.array(
            [
                [0.299, 0.587, 0.114],
                [-0.14713, -0.28886, 0.436],
                [0.615, -0.51499, -0.10001],
            ]
        )
        img_arr = img_arr.dot(transform_matrix.T)

    h, w, c = img_arr.shape
    window_size = min(window_size, h, w)
    pos_list = [
        (min(int(u * window_size / 32), window_size - 1), min(int(v * window_size / 32), window_size - 1))
        for u, v in pos_list
    ]

    for ch in channel_list:
        if ch >= c:
            continue

        channel_data = img_arr[:, :, ch]

        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                block_h = min(window_size, h - i)
                block_w = min(window_size, w - j)
                block = channel_data[i : i + block_h, j : j + block_w]

                if block_h != window_size or block_w != window_size:
                    continue

                dct_block = _dct2(block)

                for pos in pos_list:
                    u, v = pos
                    if u < window_size and v < window_size:
                        dct_block[u, v] += magnitude

                channel_data[i : i + window_size, j : j + window_size] = _idct2(dct_block)

        img_arr[:, :, ch] = channel_data

    if use_yuv:
        inverse_matrix = np.linalg.inv(transform_matrix)
        img_arr = img_arr.dot(inverse_matrix.T)

    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)


def patch_wanet(
    img: Image.Image,
    k: int = 16,
    s: float = 8.0,
    grid_rescale: float = 0.95,
    seed: int = 42,
) -> Image.Image:
    """TCAP wanet: fixed invisible warping field (requires torch)."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    x = to_tensor(img).unsqueeze(0)
    _, _, H, W = x.shape

    state = torch.get_rng_state()
    torch.manual_seed(seed)

    ins = torch.rand(1, 2, k, k) * 2 - 1

    ins = ins / torch.mean(torch.abs(ins))

    noise_grid = F.interpolate(ins, size=(H, W), mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)

    torch.set_rng_state(state)
    array_h = torch.linspace(-1, 1, steps=H)
    array_w = torch.linspace(-1, 1, steps=W)

    xx, yy = torch.meshgrid(array_h, array_w, indexing="ij")
    identity_grid = torch.stack((yy, xx), 2).unsqueeze(0)
    grid_temps = (identity_grid + s * noise_grid / H) * grid_rescale

    grid_temps = torch.clamp(grid_temps, -1, 1)
    x_bd = F.grid_sample(x, grid_temps, align_corners=True)
    output_pil = to_pil(x_bd.squeeze(0))

    return output_pil


def apply_visual_trigger(img: Image.Image, visual_trigger: str, **params) -> Image.Image:
    if visual_trigger == "black_square":
        return patch_black_square(img, **params)
    if visual_trigger == "badnet":
        return patch_badnet(img, **params)
    if visual_trigger == "blended":
        return patch_blended(img, **params)
    if visual_trigger == "sig":
        return patch_sig(img)
    if visual_trigger == "ftrojan":
        return patch_ftrojan(img)
    if visual_trigger == "wanet":
        return patch_wanet(img)
    raise ValueError(f"Unknown visual trigger: {visual_trigger}")


def _dct2(block: np.ndarray) -> np.ndarray:
    from scipy.fftpack import dct

    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    from scipy.fftpack import idct

    return idct(idct(block.T, norm="ortho").T, norm="ortho")

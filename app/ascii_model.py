from __future__ import annotations
import io
import os
import base64
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2


def _to_grayscale(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        return img.convert("L")
    return img


def _pil_from_bytes(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content)).convert("RGB")


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _tensor_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class InferenceResult:
    ascii_text: str
    output_image: Image.Image
    png_base64: str
    cols: int
    rows: int


class GlyphAtlas:
    def __init__(self, charset: str = "@%#*+=-:. ", cell_size: int = 10, font_path: str | None = None):
        self.charset = charset
        self.cell_size = cell_size
        self.font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, cell_size)
        self.templates = self._render_templates()

    def _render_templates(self) -> torch.Tensor:
        rendered: List[np.ndarray] = []
        for ch in self.charset:
            img = Image.new("L", (self.cell_size, self.cell_size), color=0)
            drawer = ImageDraw.Draw(img)
            bbox = drawer.textbbox((0, 0), ch, font=self.font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = max(0, (self.cell_size - w) // 2)
            y = max(0, (self.cell_size - h) // 2)
            drawer.text((x, y), ch, fill=255, font=self.font)
            rendered.append(np.asarray(img, dtype=np.float32) / 255.0)
        arr = np.stack(rendered, axis=0)
        return torch.from_numpy(arr)


class PatchEncoder(nn.Module):
    def __init__(self, num_symbols: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_symbols, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        logits = self.head(h)
        return logits


class AsciiArtModel:
    DEFAULT_PALETTE = (
        " `"                # brightest
        "..··∙˙°˚"         # very light
        "'`,:;¨"           # light
        "-_‾¯ˉ"            # light-medium
        "~^\"=+*"          # medium
        "il!|/\\()"        # medium-dark (escaped backslash)
        "vcxzoO0"          # dark
        "X#%&@$"           # darker
    )

    def __init__(self, model_save_path: str, cell_size: int = 10, charset: str | None = None, device: str | None = None):
        self.save_path = model_save_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cell_size = cell_size
        self.charset = charset or self.DEFAULT_PALETTE
        self.atlas = GlyphAtlas(charset=self.charset, cell_size=cell_size)
        self.num_symbols = len(self.atlas.charset)
        self.net = PatchEncoder(self.num_symbols).to(self.device)
        self._maybe_load()

    def _maybe_load(self) -> None:
        if os.path.exists(self.save_path):
            try:
                state = torch.load(self.save_path, map_location=self.device)
                self.net.load_state_dict(state)
            except Exception:
                pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(self.net.state_dict(), self.save_path)

    def _downsample_to_grid(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_rows = 120
        if (h // self.cell_size) > target_rows:
            scale = (target_rows * self.cell_size) / max(1, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            w, h = img.size
        w2 = (w // self.cell_size) * self.cell_size
        h2 = (h // self.cell_size) * self.cell_size
        if w2 < self.cell_size or h2 < self.cell_size:
            scale = max(self.cell_size / max(1, w), self.cell_size / max(1, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            w, h = img.size
            w2 = (w // self.cell_size) * self.cell_size
            h2 = (h // self.cell_size) * self.cell_size
        if (w2, h2) != (w, h):
            img = img.crop((0, 0, w2, h2))
        return img

    def _cv_importance(self, gray_np: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray_np, 60, 180)
        gx = cv2.Sobel(gray_np, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_np, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
        lap = cv2.Laplacian(gray_np, cv2.CV_32F, ksize=3)
        lap = np.abs(lap)
        lap = (lap - lap.min()) / (np.ptp(lap) + 1e-6)
        edges_f = edges.astype(np.float32) / 255.0
        imp = 0.5 * grad + 0.3 * edges_f + 0.2 * lap
        imp = cv2.GaussianBlur(imp, (3, 3), 0.8)
        imp = np.clip(imp, 0.0, 1.0)
        return imp

    def _reconstruct_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        K, gh, gw = self.atlas.templates.shape
        atlas = self.atlas.templates.to(probs.device).view(K, 1, gh, gw)
        template_means = atlas.mean(dim=(2, 3)).view(1, K, 1, 1)
        expected = (probs * template_means).sum(dim=1, keepdim=True)
        return expected

    def _grid_to_ascii(self, logits: torch.Tensor) -> Tuple[str, int, int]:
        idx = torch.argmax(logits, dim=1)[0]
        rows, cols = idx.shape
        text_rows: List[str] = []
        for r in range(rows):
            chars = [self.atlas.charset[int(idx[r, c])] for c in range(cols)]
            text_rows.append("".join(chars))
        return "\n".join(text_rows), cols, rows

    def _render_ascii_image(self, ascii_text: str, scale: int = 1) -> Image.Image:
        lines = ascii_text.splitlines()
        if not lines:
            return Image.new("L", (10, 10), color=255)
        font = self.atlas.font
        cell = self.cell_size * max(1, scale)
        width = max(len(line) for line in lines) * cell
        height = len(lines) * cell
        img = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(img)
        y = 0
        for line in lines:
            x = 0
            for ch in line:
                draw.text((x, y), ch, fill=0, font=font)
                x += cell
            y += cell
        return img

    def infer(self, image_bytes: bytes, cv_guidance: bool = True, guidance_strength: float = 2.0, topk_edge_chars: int = 3, mode: str = "palette") -> InferenceResult:
        img = _pil_from_bytes(image_bytes)
        img = _to_grayscale(img)
        img = self._downsample_to_grid(img)
        # Palette mode: deterministic density mapping (no background glyphs)
        if mode == "palette":
            gray = np.asarray(img, dtype=np.uint8)
            h, w = gray.shape
            cs = self.cell_size
            rows = h // cs
            cols = w // cs
            lines: List[str] = []
            for r in range(rows):
                row_chars: List[str] = []
                rs, re = r * cs, (r + 1) * cs
                for c in range(cols):
                    csx, cex = c * cs, (c + 1) * cs
                    patch = gray[rs:re, csx:cex]
                    mean = float(patch.mean())
                    inv = 255.0 - mean
                    idx = int(round(inv / 255.0 * (self.num_symbols - 1)))
                    idx = max(0, min(self.num_symbols - 1, idx))
                    row_chars.append(self.atlas.charset[idx])
                lines.append("".join(row_chars))
            ascii_text = "\n".join(lines)
            out_img = self._render_ascii_image(ascii_text)
            b64 = _b64_png(out_img)
            return InferenceResult(ascii_text=ascii_text, output_image=out_img, png_base64=b64, cols=cols, rows=rows)
        tens = _image_to_tensor(img).to(self.device)
        with torch.no_grad():
            logits = self.net(tens)
        if cv_guidance:
            gray_np = np.asarray(img, dtype=np.uint8)
            imp = self._cv_importance(gray_np)
            K, gh, gw = self.atlas.templates.shape
            template_means = self.atlas.templates.mean(dim=(1, 2)).cpu().numpy()
            desired_dark = 1.0 - imp
            prior = -np.abs(template_means.reshape(1, 1, K) - desired_dark[:, :, None])
            prior = prior / (np.abs(prior).mean() + 1e-6)
            prior_t = torch.from_numpy(prior).permute(2, 0, 1).unsqueeze(0).to(logits.device)
            logits = logits + guidance_strength * prior_t
            edge_mask = (imp > 0.6)
            if edge_mask.any():
                darkest_idx = np.argsort(template_means)[:max(1, topk_edge_chars)]
                mask_t = torch.from_numpy(edge_mask.astype(np.float32)).to(logits.device).unsqueeze(0).unsqueeze(0)
                boost = torch.zeros_like(logits)
                boost[:, darkest_idx, :, :] = 1.0
                logits = logits + mask_t * (0.8 * guidance_strength) * boost
        ascii_text, cols, rows = self._grid_to_ascii(logits)
        out_img = self._render_ascii_image(ascii_text)
        b64 = _b64_png(out_img)
        return InferenceResult(ascii_text=ascii_text, output_image=out_img, png_base64=b64, cols=cols, rows=rows)

    # Self-supervised online learning from a single image using reconstruction loss
    def online_train(self, image_bytes: bytes, steps: int = 150, lr: float = 1e-3) -> None:
        img = _to_grayscale(_pil_from_bytes(image_bytes))
        img = self._downsample_to_grid(img)
        target = _image_to_tensor(img).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for _ in range(steps):
            opt.zero_grad()
            logits = self.net(target)
            recon = self._reconstruct_from_logits(logits)
            loss = F.mse_loss(recon, target)
            loss.backward()
            opt.step()
        self.net.eval()
        self._save()

    # Supervised training with input/target pair and basic augmentations
    def supervised_train_pair(self, input_bytes: bytes, target_bytes: bytes, epochs: int = 3, lr: float = 1e-3, aug_per_epoch: int = 6) -> None:
        src_img = _to_grayscale(_pil_from_bytes(input_bytes))
        tgt_img = _to_grayscale(_pil_from_bytes(target_bytes))
        src_img = self._downsample_to_grid(src_img)
        tgt_img = tgt_img.resize(src_img.size, Image.BICUBIC)
        def best_symbol_indices(target_img: Image.Image) -> torch.Tensor:
            arr = np.asarray(target_img, dtype=np.float32) / 255.0
            H, W = arr.shape
            atlas = self.atlas.templates.numpy()
            means = atlas.mean(axis=(1, 2))
            desired = arr
            d = np.abs(means.reshape(1, 1, -1) - (1.0 - desired)[:, :, None])
            idx = np.argmin(d, axis=2).astype(np.int64)
            return torch.from_numpy(idx)
        def apply_augs(img_src: Image.Image, img_tgt: Image.Image) -> Tuple[Image.Image, Image.Image]:
            s, t = img_src, img_tgt
            if np.random.rand() < 0.5:
                s = s.transpose(Image.FLIP_LEFT_RIGHT)
                t = t.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.25:
                s = s.transpose(Image.FLIP_TOP_BOTTOM)
                t = t.transpose(Image.FLIP_TOP_BOTTOM)
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-5, 5)
                s = s.rotate(angle, resample=Image.BICUBIC)
                t = t.rotate(angle, resample=Image.BICUBIC)
            s = self._downsample_to_grid(s)
            t = t.resize(s.size, Image.BICUBIC)
            return s, t
        self.net.train()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for _ in range(aug_per_epoch):
                s_aug, t_aug = apply_augs(src_img, tgt_img)
                x = _image_to_tensor(s_aug).to(self.device)
                with torch.no_grad():
                    y_idx = best_symbol_indices(t_aug).to(self.device)
                logits = self.net(x)
                loss = loss_fn(logits[0], y_idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        self._save()



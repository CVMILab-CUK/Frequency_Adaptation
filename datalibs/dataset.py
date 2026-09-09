from typing import Optional
import json
import os

import cv2
import numpy as np
import torch

from .compose import Frequency_Filtering  # re-exported for convenience


class FFHQDataset(torch.utils.data.Dataset):
    """FFHQ with optional per-image captions.

    A split file is a newline- or comma-separated list of image filenames
    living under `img_dir`. If `caption_path` is given (or a `captions.jsonl`
    sits next to the images) each sample carries its own caption, which is what
    makes CLIP-score a meaningful metric -- a single hard-coded prompt for all
    70k images makes text alignment untestable.

    Returns per sample:
        gt              (3, H, W) float32 in [-1, 1]
        filtered_image  (3, H, W) float32 in [-1, 1]
        cutoff          scalar float32, the normalised radius actually used
        caption         str
        index           int
    """

    def __init__(self,
                 data_dir: str,
                 img_dir: str,
                 mode: Optional[str] = "train",
                 transforms=None,
                 caption_path: Optional[str] = None,
                 default_caption: str = "a photo of a face"):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.transform = transforms
        self.default_caption = default_caption

        if mode not in ("train", "valid", "test"):
            raise ValueError(f"mode must be train/valid/test, got {mode!r}")
        self.data_path = os.path.join(self.data_dir, f"{mode}.txt")

        with open(self.data_path, "r") as f:
            raw = f.read()
        self.lst_data = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
        if not self.lst_data:
            raise RuntimeError(f"split file {self.data_path} is empty")

        self.captions = {}
        if caption_path is None:
            guess = os.path.join(os.path.dirname(img_dir.rstrip("/")), "captions.jsonl")
            caption_path = guess if os.path.exists(guess) else None
        if caption_path and os.path.exists(caption_path):
            with open(caption_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self.captions[rec["file"]] = rec.get("caption", "") or ""

        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        name = str(self.lst_data[index])
        path = os.path.join(self.img_dir, name)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read image {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.dtype == np.uint8:
            img = img / 255.0

        data = {
            "gt": img.astype(np.float32),
            "filtered_image": img.copy().astype(np.float32),
        }

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)
        data["caption"] = self.captions.get(name, self.default_caption) or self.default_caption
        data["index"] = index
        return data


class ToTensor(object):
    """(H, W, C) numpy -> (C, H, W) torch, leaving non-array entries alone."""

    ARRAY_KEYS = ("gt", "filtered_image")

    def __call__(self, data):
        out = {}
        for key, value in data.items():
            if key in self.ARRAY_KEYS:
                out[key] = torch.from_numpy(np.ascontiguousarray(
                    value.transpose((2, 0, 1)).astype(np.float32)))
            elif isinstance(value, np.ndarray) or np.isscalar(value):
                out[key] = torch.as_tensor(np.array(value, dtype=np.float32))
            else:
                out[key] = value
        return out

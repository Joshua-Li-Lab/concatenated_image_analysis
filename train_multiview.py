"""
train_multiview.py

Combined model / data / augmentation / metric / training-support module for the
multi-stain fusion benchmark.
"""
from __future__ import annotations
import os
import copy
import hashlib
import json
import math
import queue
import random
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


VERSION = "2026-09-11"


# =====================================================================
# Constants
# =====================================================================

PATHS = {
    "bank_dir": "path",
    "output_dir": "path",
}

ALL_STAINS = ["HE", "TTF1", "MNF116", "p53", "MIB1"]
REFERENCE_STAIN = "HE"

TYPES = ["normal", "cancer"]
LABEL_MAP = {"normal": 0, "cancer": 1}
KEY_COLS = ["slide", "indication", "row", "col"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PLAIN_MEAN = (0.5, 0.5, 0.5)
PLAIN_STD = (0.5, 0.5, 0.5)

DEFAULT_BACKBONE = "timm:resnet10t"
N_REF_STAINS = len(ALL_STAINS)
OD_FLOOR = 1.0 / 255.0

PER_CELL_STD_FLOOR_LEGACY = 1e-5
PER_CELL_STD_FLOOR_ONE_LEVEL = 1.0 / 255.0
DEFAULT_PER_CELL_STD_FLOOR = PER_CELL_STD_FLOOR_LEGACY

USE_GROUPNORM_2D = True
GROUPNORM_TARGET_GROUPS = 8


def _gn_groups(channels: int, target: int = GROUPNORM_TARGET_GROUPS) -> int:
    channels = int(channels)
    for groups in range(min(int(target), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def convert_bn_to_gn(module: nn.Module,
                     target_groups: int = GROUPNORM_TARGET_GROUPS) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            channels = int(child.num_features)
            replacement = nn.GroupNorm(
                _gn_groups(channels, target_groups), channels,
                eps=float(child.eps), affine=True,
            )
            if child.affine and child.weight is not None:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight.detach())
                    replacement.bias.copy_(child.bias.detach())
            setattr(module, name, replacement)
        else:
            convert_bn_to_gn(child, target_groups)
    return module


def count_batchnorm(module: nn.Module) -> int:
    return sum(
        1 for layer in module.modules()
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d,
                              nn.BatchNorm3d, nn.SyncBatchNorm))
    )

DEFAULT_SIGNED_BORDER_DIST_PX = 0.0


def signed_border_dist_px(index: pd.DataFrame) -> pd.Series:
    if "dist_border_um" not in index.columns:
        return pd.Series(np.full(len(index), np.nan), index=index.index)
    um = pd.to_numeric(index["dist_border_um"], errors="coerce")
    if "umpp" in index.columns:
        umpp = pd.to_numeric(index["umpp"], errors="coerce").replace(0.0, np.nan)
    else:
        umpp = np.nan
    return um / umpp


def filter_by_border(index: pd.DataFrame,
                     min_signed_border_dist_px: float = DEFAULT_SIGNED_BORDER_DIST_PX,
                     keep_unknown_when_nonpositive: bool = True) -> pd.DataFrame:
    if "dist_border_um" not in index.columns:
        return index
    thr = float(min_signed_border_dist_px)
    dpx = signed_border_dist_px(index)
    keep = dpx.notna() & (dpx >= thr)
    if keep_unknown_when_nonpositive and thr <= 0.0:
        keep = keep | dpx.isna()
    return index[keep]


DEFAULT_SIGNED_BORDER_DIST_UM = 0.0


def signed_border_dist_um(index: pd.DataFrame) -> pd.Series:
    if "dist_border_um" not in index.columns:
        return pd.Series(np.full(len(index), np.nan), index=index.index)
    return pd.to_numeric(index["dist_border_um"], errors="coerce")


def filter_by_border_um(index: pd.DataFrame,
                        min_signed_border_dist_um: float = DEFAULT_SIGNED_BORDER_DIST_UM,
                        keep_unknown_when_nonpositive: bool = True) -> pd.DataFrame:
    if "dist_border_um" not in index.columns:
        return index
    thr = float(min_signed_border_dist_um)
    dum = signed_border_dist_um(index)
    keep = dum.notna() & (dum >= thr)
    if keep_unknown_when_nonpositive and thr <= 0.0:
        keep = keep | dum.isna()
    return index[keep]


# --- stack3d / colour constants ---------------------------------------
STACK3D_KERNEL_DEPTH = 3
STACK3D_STEM_STRIDE = 4

OD_CEIL = 20.0
STAIN_ROBUST_FLOOR = 0.15
STAIN_SCALE_MIN = 0.1
STAIN_SCALE_MAX = 10.0

_HED = np.array(
    [[0.650, 0.704, 0.286],
     [0.072, 0.990, 0.105],
     [0.268, 0.570, 0.776]],
    dtype=np.float64,
)
_HED = _HED / np.linalg.norm(_HED, axis=1, keepdims=True)
_HED_INV = np.linalg.inv(_HED)

STAIN_REF_MAX = (1.0, 1.0, 1.0)

TTA_VIEWS = [0, 1, 4, 5]
COLOR_MODES = ("rgb", "od_norm", "gray", "hed_std", "stain_norm")


# =====================================================================
# Reproducibility / device
# =====================================================================

def set_seed(seed: int, strict: bool = False) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(strict)
        torch.backends.cudnn.benchmark = not bool(strict)
    try:
        torch.use_deterministic_algorithms(bool(strict), warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(bool(strict))
    except Exception:
        if strict:
            raise


def get_device(prefer: str = "auto") -> torch.device:
    prefer = str(prefer).lower().strip()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if prefer in ("auto", "mps") and mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype_for(device: torch.device, use_amp: bool) -> Tuple[bool, torch.dtype]:
    if not use_amp or device.type == "cpu":
        return False, torch.float32
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        return True, torch.float16
    if device.type == "mps":
        return False, torch.float32
    return False, torch.float32


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


# =====================================================================
# Patient / case utilities
# =====================================================================

def _stem(name: str) -> str:
    base = os.path.basename(str(name))
    base = re.sub(r"\.npy$", "", base, flags=re.IGNORECASE)
    if base.startswith("bank_"):
        base = base[len("bank_"):]
    return base


def case_of(name: str) -> str:
    """Case/accession id = everything before the first hyphen ('24BX32764-A3'
    -> '24BX32764'). Used only for leakage auditing / reporting; folds are
    split at the slide level."""
    return _stem(name).split("-", 1)[0]


def slide_of(name: str) -> str:
    parts = _stem(name).split("-", 1)
    return parts[1] if len(parts) > 1 else ""


def pos_key(slide, indication, row, col) -> str:
    return f"{slide}|{indication}|{row}|{col}"


def key_series(df: pd.DataFrame) -> pd.Series:
    return df[KEY_COLS].astype(str).agg("|".join, axis=1)


# =====================================================================
# Arms
# =====================================================================

def grid_dims(k: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(max(int(k), 1))))
    rows = int(math.ceil(int(k) / cols))
    return rows, cols


@dataclass
class ArmSpec:
    name: str
    mode: str
    stains: List[str]
    readout: str = "gap"
    note: str = ""

    @property
    def k(self) -> int:
        return len(self.stains)

    def image_grid(self) -> Tuple[int, int]:
        if self.mode == "vstrip":
            return self.k, 1
        if self.mode == "grid":
            return grid_dims(self.k)
        return 1, 1

    def n_groups(self) -> int:
        if self.readout == "gap":
            return 1
        if self.mode in ("vstrip", "grid", "perstain"):
            return self.k
        return 1

    def in_channels(self) -> int:
        return 3 * self.k if self.mode == "stack" else 3

    def occupied_cells(self) -> List[Tuple[int, int]]:
        if self.mode not in ("vstrip", "grid") or self.readout == "gap":
            return [(0, 0)]
        _, cols = self.image_grid()
        return [divmod(index, cols) for index in range(self.k)]


def build_arm_library() -> Dict[str, ArmSpec]:
    arms: Dict[str, ArmSpec] = {}
    five = list(ALL_STAINS)
    ihc = [s for s in ALL_STAINS if s != REFERENCE_STAIN]

    def add(spec: ArmSpec) -> None:
        if spec.name in arms:
            raise ValueError(f"Duplicate arm name: {spec.name}")
        arms[spec.name] = spec

    # Single-stain baselines.
    for stain in five:
        add(ArmSpec(f"single_{stain}", "single", [stain], "gap",
                    "Single-stain baseline"))

    # HE + one IHC (vertical strip, global average pooling).
    for stain in ihc:
        add(ArmSpec(f"concat_v2_HE_{stain}", "vstrip",
                    [REFERENCE_STAIN, stain], "gap"))

    # HE + TTF1 with per-band (identity-aware) readout.
    add(ArmSpec("concat_v2_HE_TTF1_cell", "vstrip",
                [REFERENCE_STAIN, "TTF1"], "cell"))

    # Five-stain vertical strips.
    add(ArmSpec("concat_v5", "vstrip", five, "gap",
                "Vertical 5-stain strip, global average pooling"))
    add(ArmSpec("concat_v5_cell", "vstrip", five, "cell",
                "Vertical 5-stain strip, per-band (identity-aware) readout"))

    # 3D trunk over the stain axis.
    add(ArmSpec("stack3d", "stack3d", five, "gap",
                "5-stain 3D trunk (NOT the same as stack5)"))
    add(ArmSpec("stack3d_HE_TTF1", "stack3d", [REFERENCE_STAIN, "TTF1"], "gap"))

    # Compute-matched control for concat_v5.
    add(ArmSpec("ctrl_v_HEx5", "vstrip", [REFERENCE_STAIN] * 5, "gap",
                "Compute-matched control for concat_v5"))

    return arms


ARM_LIBRARY = build_arm_library()


# =====================================================================
# Tile bank and iterators
# =====================================================================

class TileBank:
    def __init__(self, bank_dir: str, preload: bool = False):
        self.bank_dir = os.path.abspath(os.path.expanduser(bank_dir))
        metadata_path = os.path.join(self.bank_dir, "bank_meta.json")
        index_path = os.path.join(self.bank_dir, "bank_index.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(metadata_path)
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)

        with open(metadata_path, "r", encoding="utf-8") as handle:
            self.meta = json.load(handle)

        if not self.meta.get("sharded", False):
            raise ValueError("bank_meta.json must contain 'sharded': true.")

        self.size = int(self.meta["size"])
        self.stains = list(self.meta["stains"])
        self.stain_idx = {stain: i for i, stain in enumerate(self.stains)}

        self.index = pd.read_csv(index_path,
                                 dtype={"slide": str, "row": str, "col": str})

        required = {"bank_idx", "slide", "local_idx", "indication", "row", "col"}
        missing = required.difference(self.index.columns)
        if missing:
            raise KeyError(f"bank_index.csv missing: {sorted(missing)}")

        self.index = (self.index.sort_values("bank_idx", kind="stable")
                      .reset_index(drop=True))

        observed = self.index["bank_idx"].astype(np.int64).to_numpy()
        expected = np.arange(len(self.index), dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise ValueError("bank_idx must be contiguous and zero-based.")

        self.row_slide = self.index["slide"].astype(str).to_numpy()
        self.row_local = self.index["local_idx"].astype(np.int64).to_numpy()

        self.key_to_row = {
            pos_key(r.slide, r.indication, r.row, r.col): int(r.bank_idx)
            for r in self.index.itertuples()
        }

        self.shards: Dict[str, np.ndarray] = {}
        for slide, shard_info in self.meta.get("shards", {}).items():
            slide = str(slide)
            filename = str(shard_info["file"])
            if os.path.basename(filename).startswith("._"):
                continue
            path = os.path.join(self.bank_dir, filename)
            if not os.path.exists(path):
                print(f"[WARNING] Missing shard {slide}: {path}")
                continue
            self.shards[slide] = (np.load(path) if preload
                                  else np.load(path, mmap_mode="r"))

        if not self.shards:
            raise RuntimeError("No readable shard files were loaded.")

        first = next(iter(self.shards.values()))
        if first.ndim != 5 or first.shape[-1] != 3:
            raise ValueError("Shard shape must be (positions, stains, H, W, 3).")

        self._n_stains = int(first.shape[1])
        if self._n_stains != len(self.stains):
            raise ValueError("Shard stain dimension differs from metadata.")

        print(f"Tile bank: {len(self.index):,} positions, "
              f"{len(self.stains)} stains, {self.size}px, "
              f"{len(self.shards)} shards ({'RAM' if preload else 'mmap'})")

    @property
    def arr_shape0(self) -> int:
        return len(self.index)

    def gather(self, pos_idx: np.ndarray, stain_ids: Sequence[int]) -> np.ndarray:
        pos_idx = np.asarray(pos_idx, dtype=np.int64)
        stain_ids = np.asarray(stain_ids, dtype=np.int64)

        if np.any(pos_idx < 0) or np.any(pos_idx >= len(self.index)):
            raise IndexError("Position outside tile bank.")
        if np.any(stain_ids < 0) or np.any(stain_ids >= self._n_stains):
            raise IndexError("Stain outside tile bank.")

        output = np.empty((len(pos_idx), len(stain_ids), self.size, self.size, 3),
                          dtype=np.uint8)

        slides = self.row_slide[pos_idx]
        local_indices = self.row_local[pos_idx]

        for slide in np.unique(slides):
            mask = slides == slide
            destination = np.flatnonzero(mask)
            shard = self.shards.get(str(slide))
            if shard is None:
                raise FileNotFoundError(f"No shard for slide {slide}")
            local = local_indices[mask]
            if np.any(local < 0) or np.any(local >= shard.shape[0]):
                raise IndexError(f"local_idx outside shard {slide}")
            output[destination] = shard[np.ix_(local, stain_ids)]

        return np.ascontiguousarray(output)


class BatchIterator:
    def __init__(self, bank: TileBank, pos_idx: np.ndarray, labels: np.ndarray,
                 weights: np.ndarray, meta: pd.DataFrame, stain_ids: Sequence[int],
                 batch_size: int, shuffle: bool, seed: int = 0):
        self.bank = bank
        self.pos_idx = np.asarray(pos_idx, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.meta = meta.reset_index(drop=True)
        self.stain_ids = [int(i) for i in stain_ids]
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        lengths = {len(self.pos_idx), len(self.labels),
                   len(self.weights), len(self.meta)}
        if len(lengths) != 1:
            raise ValueError("Iterator arrays have different lengths.")
        if self.bs <= 0:
            raise ValueError("batch_size must be positive.")

    def __len__(self) -> int:
        return int(math.ceil(len(self.pos_idx) / self.bs))

    @property
    def n(self) -> int:
        return len(self.pos_idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = np.arange(len(self.pos_idx))
        if self.shuffle:
            rng = np.random.default_rng(self.seed * 100003 + self.epoch)
            rng.shuffle(order)
        for start in range(0, len(order), self.bs):
            selected = order[start:start + self.bs]
            images = self.bank.gather(self.pos_idx[selected], self.stain_ids)
            yield (torch.from_numpy(images),
                   torch.from_numpy(self.labels[selected].astype(np.float32)),
                   torch.from_numpy(self.weights[selected]),
                   selected)


class Prefetcher:
    def __init__(self, iterator, depth: int = 4):
        self.it = iterator
        self.depth = max(1, int(depth))

    def __len__(self) -> int:
        return len(self.it)

    @property
    def n(self) -> int:
        return self.it.n

    @property
    def meta(self) -> pd.DataFrame:
        return self.it.meta

    @property
    def labels(self) -> np.ndarray:
        return self.it.labels

    @property
    def weights(self) -> np.ndarray:
        return self.it.weights

    def set_epoch(self, epoch: int) -> None:
        self.it.set_epoch(epoch)

    def __iter__(self):
        work_queue: queue.Queue = queue.Queue(maxsize=self.depth)
        stop_event = threading.Event()
        sentinel = object()

        def put_safely(item) -> bool:
            while not stop_event.is_set():
                try:
                    work_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def worker() -> None:
            try:
                for item in self.it:
                    if stop_event.is_set():
                        break
                    if not put_safely(("item", item)):
                        return
            except BaseException as exc:
                put_safely(("error", exc))
            finally:
                put_safely(("end", sentinel))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                kind, payload = work_queue.get()
                if kind == "item":
                    yield payload
                elif kind == "error":
                    raise payload
                elif kind == "end":
                    break
        finally:
            stop_event.set()
            thread.join(timeout=2.0)


# =====================================================================
# GPU preprocessing
# =====================================================================

@dataclass
class AugConfig:
    dihedral: bool = True
    jitter: float = 0.10
    jitter_p: float = 0.8
    hed_sigma: float = 0.08
    hed_bias: float = 0.03
    hed_p: float = 0.9
    gray_p: float = 0.15
    stain_dropout: float = 0.25
    cutout_p: float = 0.25
    cutout_frac: float = 1.0 / N_REF_STAINS


def _dihedral(tensor: torch.Tensor, transform: int) -> torch.Tensor:
    transform = int(transform)
    output = (torch.rot90(tensor, transform % 4, dims=(-2, -1))
              if transform % 4 else tensor)
    if transform >= 4:
        output = torch.flip(output, dims=[-1])
    return output


class GPUPipeline:
    """Colour / normalisation pipeline. In eval mode every op is independent
    per (position, stain) plane, so one cached tensor is reusable by every arm."""

    def __init__(self, device: torch.device, tile_size: int, bank_size: int,
                 normalise: str = "imagenet", per_cell_standardize: bool = True,
                 color_mode: str = "rgb",
                 per_cell_std_floor: float = DEFAULT_PER_CELL_STD_FLOOR):
        if color_mode not in COLOR_MODES:
            raise ValueError(f"Unknown color mode: {color_mode}")
        if not float(per_cell_std_floor) > 0.0:
            raise ValueError("per_cell_std_floor must be positive.")

        self.device = device
        self.S = int(tile_size)
        self.bank_size = int(bank_size)
        self.per_cell_std = bool(per_cell_standardize)
        self.color_mode = color_mode
        self.per_cell_std_floor = float(per_cell_std_floor)

        mean, std = ((IMAGENET_MEAN, IMAGENET_STD) if normalise == "imagenet"
                     else (PLAIN_MEAN, PLAIN_STD))
        self.mean = torch.tensor(mean, dtype=torch.float32,
                                 device=device).reshape(1, 1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32,
                                device=device).reshape(1, 1, 3, 1, 1)

        self.M = torch.tensor(_HED, dtype=torch.float32, device=device)
        self.Minv = torch.tensor(_HED_INV, dtype=torch.float32, device=device)
        self.stain_ref_max = torch.tensor(STAIN_REF_MAX, dtype=torch.float32,
                                          device=device)
        self.luminance = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32,
                                      device=device).reshape(1, 1, 3, 1, 1)

    def _to_hed(self, images: torch.Tensor) -> torch.Tensor:
        optical_density = -torch.log(images.clamp_min(OD_FLOOR))
        return optical_density.permute(0, 1, 3, 4, 2) @ self.Minv

    def _from_hed(self, concentrations: torch.Tensor) -> torch.Tensor:
        optical_density = (concentrations @ self.M).clamp(0.0, OD_CEIL)
        return torch.exp(-optical_density).permute(0, 1, 4, 2, 3)

    def _od_normalize(self, images: torch.Tensor) -> torch.Tensor:
        optical_density = -torch.log(images.clamp_min(OD_FLOOR))
        batch, stains, _, height, width = optical_density.shape
        gray_od = optical_density.mean(2).reshape(batch * stains, height * width)
        top_count = max(1, (height * width) // 20)
        scale = gray_od.topk(top_count, dim=1).values.mean(1).clamp_min(0.05)
        optical_density = optical_density / scale.reshape(batch, stains, 1, 1, 1)
        return torch.exp(-optical_density.clamp(0, OD_CEIL)).clamp(0.0, 1.0)

    def _stain_normalize(self, images: torch.Tensor) -> torch.Tensor:
        concentrations = self._to_hed(images)
        batch, stains, height, width, channels = concentrations.shape
        flat = concentrations.clamp_min(0.0).reshape(
            batch, stains, height * width, channels)
        top_count = max(1, (height * width) // 100)
        robust_max = (flat.topk(top_count, dim=2).values.mean(2)
                      ).clamp_min(STAIN_ROBUST_FLOOR)
        scale = (self.stain_ref_max.reshape(1, 1, channels) / robust_max
                 ).clamp(STAIN_SCALE_MIN, STAIN_SCALE_MAX)
        concentrations = concentrations * scale.reshape(
            batch, stains, 1, 1, channels)
        return self._from_hed(concentrations)

    def __call__(self, x_u8: torch.Tensor, train: bool, aug: AugConfig,
                 rng=None, tta_view: int = 0) -> torch.Tensor:
        del rng
        images = (x_u8.to(self.device, non_blocking=True)
                  .permute(0, 1, 4, 2, 3).float().div_(255.0))
        batch, stains = images.shape[:2]

        if self.S != self.bank_size:
            images = F.interpolate(
                images.reshape(batch * stains, 3, self.bank_size, self.bank_size),
                size=(self.S, self.S), mode="bilinear",
                align_corners=False, antialias=True,
            ).reshape(batch, stains, 3, self.S, self.S)

        if train and aug.dihedral:
            random_ops = torch.rand(3, batch, 1, 1, 1, 1, device=self.device) < 0.5
            images = torch.where(random_ops[0], images.transpose(-1, -2), images)
            images = torch.where(random_ops[1], torch.flip(images, dims=[-1]), images)
            images = torch.where(random_ops[2], torch.flip(images, dims=[-2]), images)
        elif not train and int(tta_view) != 0:
            images = _dihedral(images, int(tta_view))

        if train and aug.hed_sigma > 0 and aug.hed_p > 0:
            active = (torch.rand(batch, stains, 1, 1, 1, device=self.device)
                      < aug.hed_p).float()
            scale = 1.0 + (torch.rand(batch, stains, 1, 1, 3, device=self.device)
                           * 2.0 - 1.0) * aug.hed_sigma * active
            bias = (torch.rand(batch, stains, 1, 1, 3, device=self.device)
                    * 2.0 - 1.0) * aug.hed_bias * active
            concentrations = self._to_hed(images)
            images = self._from_hed(concentrations * scale + bias)

        if train and aug.jitter > 0:
            active = (torch.rand(batch, stains, 1, 1, 1, device=self.device)
                      < aug.jitter_p).float()
            brightness = (torch.rand(batch, stains, 1, 1, 1, device=self.device)
                          * 2.0 - 1.0) * aug.jitter * active
            contrast = 1.0 + (torch.rand(batch, stains, 1, 1, 1, device=self.device)
                              * 2.0 - 1.0) * aug.jitter * active
            mean = images.mean(dim=(2, 3, 4), keepdim=True)
            images = ((images - mean) * contrast + mean + brightness).clamp_(0.0, 1.0)

        if train and aug.gray_p > 0:
            active = (torch.rand(batch, 1, 1, 1, 1, device=self.device)
                      < aug.gray_p).float()
            gray = (images * self.luminance).sum(2, keepdim=True).expand_as(images)
            images = images * (1.0 - active) + gray * active

        if self.color_mode == "od_norm":
            images = self._od_normalize(images)
        elif self.color_mode == "stain_norm":
            images = self._stain_normalize(images)
        elif self.color_mode == "gray":
            images = ((images * self.luminance).sum(2, keepdim=True)
                      .expand(-1, -1, 3, -1, -1))
        elif self.color_mode == "hed_std":
            concentrations = self._to_hed(images).permute(0, 1, 4, 2, 3)
            mean = concentrations.mean(dim=(3, 4), keepdim=True)
            std = concentrations.std(dim=(3, 4), keepdim=True,
                                     correction=0).clamp_min(self.per_cell_std_floor)
            return torch.nan_to_num(((concentrations - mean) / std).contiguous())

        if self.per_cell_std:
            mean = images.mean(dim=(3, 4), keepdim=True)
            std = images.std(dim=(3, 4), keepdim=True,
                             correction=0).clamp_min(self.per_cell_std_floor)
            images = (images - mean) / std
        else:
            images = (images - self.mean) / self.std

        return torch.nan_to_num(images.contiguous())


def modality_dropout(images: torch.Tensor, aug: AugConfig, rng=None,
                     n_ref: int = N_REF_STAINS) -> torch.Tensor:
    del rng, n_ref
    batch, stains = images.shape[:2]
    device = images.device

    if stains >= 2:
        if aug.stain_dropout <= 0:
            return images
        active = torch.rand(batch, device=device) < aug.stain_dropout
        if not bool(active.any()):
            return images
        stain_index = torch.randint(0, stains, (batch,), device=device)
        selection = torch.zeros((batch, stains), dtype=images.dtype, device=device)
        selection[torch.arange(batch, device=device), stain_index] = 1.0
        mask = 1.0 - selection * active.to(images.dtype).unsqueeze(1)
        return images * mask.reshape(batch, stains, 1, 1, 1)

    if aug.cutout_p <= 0 or aug.cutout_frac <= 0:
        return images
    height, width = images.shape[3:5]
    cut_height = min(height, max(1, int(round(height * math.sqrt(aug.cutout_frac)))))
    cut_width = min(width, max(1, int(round(width * math.sqrt(aug.cutout_frac)))))
    active = torch.rand(batch, device=device) < aug.cutout_p
    if not bool(active.any()):
        return images
    top = torch.randint(0, height - cut_height + 1, (batch,), device=device)
    left = torch.randint(0, width - cut_width + 1, (batch,), device=device)
    row = torch.arange(height, device=device).reshape(1, height, 1)
    column = torch.arange(width, device=device).reshape(1, 1, width)
    box = ((row >= top.reshape(batch, 1, 1))
           & (row < top.reshape(batch, 1, 1) + cut_height)
           & (column >= left.reshape(batch, 1, 1))
           & (column < left.reshape(batch, 1, 1) + cut_width)
           & active.reshape(batch, 1, 1))
    return images * (~box).to(images.dtype).reshape(batch, 1, 1, height, width)


# =====================================================================
# Two-dimensional backbones
# =====================================================================

class _GNBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _gn_groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation(inputs + output)


class StainNetGN(nn.Module):
    def __init__(self, in_chans: int = 3,
                 widths: Sequence[int] = (32, 64, 128, 256)):
        super().__init__()
        widths = tuple(int(width) for width in widths)
        first = widths[0]
        groups = _gn_groups(first)
        layers: List[nn.Module] = [
            nn.Conv2d(in_chans, first, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, first), nn.SiLU(inplace=True),
            nn.Conv2d(first, first, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, first), nn.SiLU(inplace=True),
            _GNBlock(first),
        ]
        channels = first
        for width in widths[1:]:
            layers.extend([
                nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_gn_groups(width), width), nn.SiLU(inplace=True),
                _GNBlock(width),
            ])
            channels = width
        self.body = nn.Sequential(*layers)
        self.out_dim = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


@torch.no_grad()
def _fix_stem_variance(module: nn.Module, in_chans: int) -> None:
    if in_chans <= 3:
        return
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d) and layer.in_channels == in_chans:
            layer.weight.mul_(math.sqrt(in_chans / 3.0))
            return


class _TimmTrunk(nn.Module):
    def __init__(self, name: str, in_chans: int, pretrained: bool):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm: pip install timm") from exc

        self.model = timm.create_model(name, pretrained=bool(pretrained),
                                        features_only=True, in_chans=int(in_chans))
        if pretrained:
            _fix_stem_variance(self.model, int(in_chans))

        channels = self.model.feature_info.channels()
        if not channels:
            raise RuntimeError(f"{name} returned no feature channels.")
        self.out_dim = int(channels[-1])

        if USE_GROUPNORM_2D:
            convert_bn_to_gn(self.model)
            remaining = count_batchnorm(self.model)
            if remaining:
                raise RuntimeError(
                    f"{name}: {remaining} BatchNorm layer(s) survived the "
                    "GroupNorm conversion; fairness would be broken.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.model(inputs)
        if not features:
            raise RuntimeError("Backbone returned empty features.")
        return features[-1]


class _ResNetTrunk(nn.Module):
    def __init__(self, name: str, in_chans: int, pretrained: bool):
        super().__init__()
        try:
            import torchvision
        except ImportError as exc:
            raise ImportError("Install torchvision.") from exc

        constructors = {"resnet18": torchvision.models.resnet18,
                        "resnet34": torchvision.models.resnet34}
        weight_classes = {"resnet18": torchvision.models.ResNet18_Weights,
                          "resnet34": torchvision.models.ResNet34_Weights}
        if name not in constructors:
            raise ValueError(f"Unsupported backbone: {name}")

        weights = weight_classes[name].IMAGENET1K_V1 if pretrained else None
        model = constructors[name](weights=weights)

        if int(in_chans) != 3:
            old = model.conv1
            new = nn.Conv2d(int(in_chans), old.out_channels, old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                repeats = int(math.ceil(int(in_chans) / 3))
                repeated = old.weight.repeat(1, repeats, 1, 1)[:, :int(in_chans)]
                new.weight.copy_(repeated / math.sqrt(int(in_chans) / 3.0))
            model.conv1 = new

        self.out_dim = int(model.fc.in_features)
        self.body = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4)

        if USE_GROUPNORM_2D:
            convert_bn_to_gn(self.body)
            remaining = count_batchnorm(self.body)
            if remaining:
                raise RuntimeError(
                    f"{name}: {remaining} BatchNorm layer(s) survived the "
                    "GroupNorm conversion; fairness would be broken.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


def build_trunk(backbone: str, in_chans: int, pretrained: bool):
    backbone = str(backbone)
    if backbone == "stainnet_gn":
        trunk = StainNetGN(in_chans=int(in_chans))
    elif backbone.startswith("timm:"):
        name = backbone.split(":", 1)[1].strip()
        if not name:
            raise ValueError("Missing timm model name.")
        trunk = _TimmTrunk(name, int(in_chans), bool(pretrained))
    elif backbone in ("resnet18", "resnet34"):
        trunk = _ResNetTrunk(backbone, int(in_chans), bool(pretrained))
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return trunk, int(trunk.out_dim)


# =====================================================================
# 3D backbone (stack3d)
# =====================================================================

class Conv3dGN(nn.Module):
    def __init__(self, cin: int, cout: int, spatial_stride: int = 1,
                 depth_stride: int = 1, kernel_depth: int = STACK3D_KERNEL_DEPTH):
        super().__init__()
        kd = int(kernel_depth)
        self.conv = nn.Conv3d(
            int(cin), int(cout), kernel_size=(kd, 3, 3),
            stride=(int(depth_stride), int(spatial_stride), int(spatial_stride)),
            padding=(kd // 2, 1, 1), bias=False)
        self.norm = nn.GroupNorm(_gn_groups(int(cout)), int(cout))
        self.activation = nn.SiLU(inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class Res3DBlock(nn.Module):
    def __init__(self, channels: int, kernel_depth: int = STACK3D_KERNEL_DEPTH):
        super().__init__()
        self.first = Conv3dGN(channels, channels, kernel_depth=kernel_depth)
        self.second = Conv3dGN(channels, channels, kernel_depth=kernel_depth)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.second(self.first(inputs)))


class Stain3DTrunk(nn.Module):
    """Compact 3D CNN over the stain axis, returning a 2D feature map."""

    def __init__(self, in_chans: int = 3, base: int = 32, out_dim: int = 512,
                 depth: int = N_REF_STAINS,
                 kernel_depth: int = STACK3D_KERNEL_DEPTH,
                 stem_stride: int = STACK3D_STEM_STRIDE):
        super().__init__()
        del depth
        width = int(base)
        widths = [width, width * 2, width * 4, width * 4]
        spatial_strides = [1, 2, 2, 2]
        depth_strides = [1, 2, 2, 2]

        if int(stem_stride) == 4:
            self.stem = nn.Sequential(
                Conv3dGN(in_chans, width, spatial_stride=2, depth_stride=1,
                         kernel_depth=kernel_depth),
                Conv3dGN(width, width, spatial_stride=2, depth_stride=1,
                         kernel_depth=kernel_depth))
        else:
            self.stem = Conv3dGN(in_chans, width, spatial_stride=2,
                                 depth_stride=1, kernel_depth=kernel_depth)

        stages: List[nn.Module] = []
        channels = width
        for out_channels, s_stride, d_stride in zip(widths, spatial_strides,
                                                     depth_strides):
            stages.append(Conv3dGN(channels, out_channels, spatial_stride=s_stride,
                                   depth_stride=d_stride, kernel_depth=kernel_depth))
            stages.append(Res3DBlock(out_channels, kernel_depth=kernel_depth))
            channels = out_channels
        self.stages = nn.Sequential(*stages)

        self.projection = nn.Conv3d(channels, int(out_dim), kernel_size=1, bias=False)
        self.projection_norm = nn.GroupNorm(_gn_groups(int(out_dim)), int(out_dim))
        self.projection_activation = nn.SiLU(inplace=False)
        self.out_dim = int(out_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.stem(inputs)
        x = self.stages(x)
        x = self.projection_activation(self.projection_norm(self.projection(x)))
        return x.mean(dim=2)


# =====================================================================
# Sparse readouts
# =====================================================================

class CrossStainReadout(nn.Module):
    KINDS = ("gap", "cell")

    def __init__(self, kind: str, n: int, dim: int, n_max: int = N_REF_STAINS,
                 rank: int = 32, heads: int = 4, vrank: int = 64,
                 dropout: float = 0.1, mixer_layers: int = 2):
        super().__init__()
        del n_max, rank, heads, vrank, mixer_layers
        if kind not in self.KINDS:
            raise ValueError(f"Unknown readout: {kind}")

        self.kind = str(kind)
        self.n = int(n)
        self.dim = int(dim)

        self.projection = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.eye(self.dim).reshape(self.dim, self.dim, 1, 1))
            self.projection.bias.zero_()

        self.film_gain = nn.Parameter(torch.ones(self.n, self.dim))
        self.film_bias = nn.Parameter(torch.zeros(self.n, self.dim))

        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        classifier_input = self.dim * self.n if kind == "cell" else self.dim
        self.classifier = nn.Linear(classifier_input, 1)

    def cells(self, cells: torch.Tensor) -> torch.Tensor:
        batch, groups, channels, height, width = cells.shape
        if groups != self.n:
            raise RuntimeError(f"Expected {self.n} groups, received {groups}.")
        if channels != self.dim:
            raise RuntimeError(f"Expected {self.dim} channels, received {channels}.")
        output = self.projection(cells.reshape(batch * groups, channels, height, width)
                                 ).reshape(batch, groups, channels, height, width)
        return (output * self.film_gain.reshape(1, groups, channels, 1, 1)
                + self.film_bias.reshape(1, groups, channels, 1, 1))

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        batch, groups, channels, height, width = cells.shape
        output = self.cells(cells)

        if self.kind == "gap":
            pooled = output.mean(dim=(1, 3, 4))
            return self.classifier(self.dropout(pooled))

        if self.kind == "cell":
            pooled = output.mean(dim=(3, 4)).reshape(batch, groups * channels)
            return self.classifier(self.dropout(pooled))

        raise RuntimeError(f"Unhandled readout: {self.kind}")


# =====================================================================
# Stack3D parameter-budget fitting
# =====================================================================

_STACK3D_CACHE: Dict[str, Tuple[int, int, int, int]] = {}


def stack3d_spec_for(backbone: str, depth: int = N_REF_STAINS,
                     base_range: Tuple[int, int, int] = (8, 160, 2),
                     readout_kind: str = "gap", stain_code: bool = True,
                     enable_aux: bool = False, ro_kw: Optional[Dict] = None,
                     kernel_depth: int = STACK3D_KERNEL_DEPTH,
                     stem_stride: int = STACK3D_STEM_STRIDE) -> Tuple[int, int]:
    """Auto-tune the 3D trunk width so stack3d matches the 2D parameter count."""
    ro_kw = dict(ro_kw or {})
    key = json.dumps(
        {"backbone": backbone, "depth": int(depth),
         "base_range": tuple(base_range), "readout_kind": readout_kind,
         "stain_code": bool(stain_code), "enable_aux": bool(enable_aux),
         "ro_kw": ro_kw, "kernel_depth": int(kernel_depth),
         "stem_stride": int(stem_stride), "gn2d": bool(USE_GROUPNORM_2D)},
        sort_keys=True, default=str)
    if key in _STACK3D_CACHE:
        base, dim, _, _ = _STACK3D_CACHE[key]
        return base, dim

    reference_trunk, feature_dim = build_trunk(backbone, in_chans=3, pretrained=False)
    target_params = sum(p.numel() for p in reference_trunk.parameters())
    del reference_trunk

    def _head_params() -> int:
        head = CrossStainReadout(
            kind=readout_kind, n=1, dim=feature_dim,
            rank=int(ro_kw.get("rank", 32)), heads=int(ro_kw.get("heads", 4)),
            vrank=int(ro_kw.get("vrank", 64)),
            dropout=float(ro_kw.get("dropout", 0.1)),
            mixer_layers=int(ro_kw.get("mixer_layers", 2)))
        n = sum(p.numel() for p in head.parameters())
        del head
        return n

    head_params = _head_params()
    target_params += head_params
    if enable_aux:
        target_params += feature_dim + 1
    if stain_code:
        target_params += 2 * 1 * 3

    fixed_params = head_params
    if enable_aux:
        fixed_params += feature_dim + 1
    if stain_code:
        fixed_params += 2 * int(depth) * 3

    low, high, step = map(int, base_range)
    if low <= 0 or high < low or step <= 0:
        raise ValueError(f"Invalid stack3d range: {base_range}")
    bases = list(range(low, high + 1, step))

    _memo: Dict[int, int] = {}

    def total_for(i: int) -> int:
        if i in _memo:
            return _memo[i]
        trunk = Stain3DTrunk(in_chans=3, base=bases[i], out_dim=feature_dim,
                             depth=int(depth), kernel_depth=kernel_depth,
                             stem_stride=stem_stride)
        total = fixed_params + sum(p.numel() for p in trunk.parameters())
        del trunk
        _memo[i] = int(total)
        return int(total)

    lo, hi = 0, len(bases) - 1
    candidates = {lo, hi}
    while hi - lo > 1:
        mid = (lo + hi) // 2
        candidates.add(mid)
        if total_for(mid) < target_params:
            lo = mid
        else:
            hi = mid
    candidates.update({lo, hi})

    best_i = min(candidates, key=lambda i: abs(total_for(i) - target_params))
    best_base, best_total = bases[best_i], total_for(best_i)
    deviation = (best_total - target_params) / max(target_params, 1) * 100.0
    print(f"[stack3d] depth={depth} base={best_base}, params={best_total:,}, "
          f"target={target_params:,}, deviation={deviation:+.2f}%, dim={feature_dim}")
    _STACK3D_CACHE[key] = (int(best_base), int(feature_dim),
                           int(best_total), int(target_params))
    return int(best_base), int(feature_dim)


# =====================================================================
# Unified model
# =====================================================================

class FusionNetBase(nn.Module):
    def __init__(self, arm: ArmSpec, backbone: str, pretrained: bool = True,
                 dropout: float = 0.1, stain_code: bool = True,
                 mixer_layers: int = 2, cell_gutter: int = 0,
                 readout_rank: int = 32, readout_heads: int = 4,
                 readout_vrank: int = 64, n_max: int = N_REF_STAINS,
                 stack3d_base_range: Tuple[int, int, int] = (8, 160, 2),
                 enable_aux: bool = False,
                 stack3d_kernel_depth: int = STACK3D_KERNEL_DEPTH,
                 stack3d_stem_stride: int = STACK3D_STEM_STRIDE):
        super().__init__()
        del n_max

        self.arm = arm
        self.mode = arm.mode
        self.k = arm.k
        self.gutter = int(cell_gutter)
        self.use_code = bool(stain_code)
        self.enable_aux = bool(enable_aux)

        readout_kwargs = {"rank": int(readout_rank), "heads": int(readout_heads),
                          "vrank": int(readout_vrank), "dropout": float(dropout),
                          "mixer_layers": int(mixer_layers)}

        if arm.mode == "stack3d":
            base, feature_dim = stack3d_spec_for(
                backbone=backbone, depth=arm.k,
                base_range=tuple(stack3d_base_range), readout_kind=arm.readout,
                stain_code=self.use_code, enable_aux=self.enable_aux,
                ro_kw=readout_kwargs, kernel_depth=int(stack3d_kernel_depth),
                stem_stride=int(stack3d_stem_stride))
            self.trunk = Stain3DTrunk(
                in_chans=3, base=base, out_dim=feature_dim, depth=arm.k,
                kernel_depth=int(stack3d_kernel_depth),
                stem_stride=int(stack3d_stem_stride))
        else:
            self.trunk, feature_dim = build_trunk(
                backbone=backbone, in_chans=arm.in_channels(),
                pretrained=bool(pretrained))

        self.feat_dim = int(feature_dim)

        if self.use_code:
            self.stain_scale = nn.Parameter(torch.ones(arm.k, 3, 1, 1))
            self.stain_shift = nn.Parameter(torch.zeros(arm.k, 3, 1, 1))
        else:
            self.register_parameter("stain_scale", None)
            self.register_parameter("stain_shift", None)

        self.head = CrossStainReadout(
            kind=arm.readout, n=arm.n_groups(), dim=self.feat_dim,
            rank=int(readout_rank), heads=int(readout_heads),
            vrank=int(readout_vrank), dropout=float(dropout),
            mixer_layers=int(mixer_layers))

        self.aux = nn.Linear(self.feat_dim, 1) if self.enable_aux else None

    def assemble(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, stains, _, size, _ = inputs.shape

        if self.mode == "single":
            return inputs[:, 0].contiguous()
        if self.mode == "perstain":
            return inputs.contiguous()
        if self.mode == "stack":
            return inputs.reshape(batch, stains * 3, size, size).contiguous()
        if self.mode == "stack3d":
            return inputs.permute(0, 2, 1, 3, 4).contiguous()

        if self.mode == "vstrip":
            if self.gutter == 0:
                return (inputs.permute(0, 2, 1, 3, 4)
                        .reshape(batch, 3, stains * size, size).contiguous())
            height = stains * size + (stains - 1) * self.gutter
            output = inputs.new_zeros(batch, 3, height, size)
            for index in range(stains):
                start = index * (size + self.gutter)
                output[:, :, start:start + size] = inputs[:, index]
            return output.contiguous()

        if self.mode == "grid":
            rows, columns = self.arm.image_grid()
            output = inputs.new_zeros(
                batch, 3, rows * size + (rows - 1) * self.gutter,
                columns * size + (columns - 1) * self.gutter)
            for index in range(stains):
                row, column = divmod(index, columns)
                row_start = row * (size + self.gutter)
                column_start = column * (size + self.gutter)
                output[:, :, row_start:row_start + size,
                       column_start:column_start + size] = inputs[:, index]
            return output.contiguous()

        raise ValueError(f"Unknown mode: {self.mode}")

    def _cells(self, assembled: torch.Tensor) -> torch.Tensor:
        if self.mode == "perstain":
            batch, stains = assembled.shape[:2]
            features = self.trunk(
                assembled.reshape(batch * stains, *assembled.shape[2:]).contiguous())
            features = features.reshape(batch, stains, *features.shape[1:])
            if self.arm.n_groups() == stains:
                return features
            return features.mean(1, keepdim=True)

        if self.mode == "stack3d":
            return self.trunk(assembled.contiguous()).unsqueeze(1)

        features = self.trunk(assembled.contiguous())
        if self.arm.n_groups() == 1:
            return features.unsqueeze(1)

        rows, columns = self.arm.image_grid()
        batch, channels, height, width = features.shape
        cell_height = max(height // rows, 1)
        cell_width = max(width // columns, 1)
        features = features[:, :, :cell_height * rows, :cell_width * columns]
        cells = (features.reshape(batch, channels, rows, cell_height,
                                  columns, cell_width)
                 .permute(0, 2, 4, 1, 3, 5)
                 .reshape(batch, rows * columns, channels, cell_height, cell_width))
        selected = [row * columns + column
                    for row, column in self.arm.occupied_cells()]
        if len(selected) < cells.shape[1]:
            cells = cells[:, selected]
        return cells.contiguous()

    def _pre(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_code:
            if self.stain_scale is None or self.stain_shift is None:
                raise RuntimeError("Stain code missing.")
            inputs = (inputs * self.stain_scale.unsqueeze(0)
                      + self.stain_shift.unsqueeze(0))
        return self._cells(self.assemble(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self._pre(inputs))

    def forward_with_aux(self, inputs: torch.Tensor):
        if not self.enable_aux or self.aux is None:
            raise RuntimeError("Auxiliary output is disabled.")
        cells = self._pre(inputs)
        logits = self.head(cells)
        auxiliary_features = self.head.cells(cells).mean(dim=(3, 4))
        auxiliary_logits = self.aux(auxiliary_features).squeeze(-1)
        return logits, auxiliary_logits


# =====================================================================
# Parameter activity / benchmark
# =====================================================================

def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def input_elements(arm: ArmSpec, tile_size: int) -> int:
    return int(3 * arm.k * int(tile_size) * int(tile_size))


def make_audit_input(arm: ArmSpec, tile_size: int, device: torch.device,
                     batch_size: int = 1) -> torch.Tensor:
    return torch.randn(int(batch_size), int(arm.k), 3, int(tile_size),
                       int(tile_size), device=device, dtype=torch.float32)


def parameter_activity(model: nn.Module, sample_input: torch.Tensor,
                       include_aux: bool = False) -> Dict:
    original_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    if include_aux:
        logits, auxiliary = model.forward_with_aux(sample_input)
        objective = logits.float().sum() + auxiliary.float().sum()
    else:
        objective = model(sample_input).float().sum()
    objective.backward()

    total = trainable = active = inactive = 0
    rows = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        is_trainable = bool(parameter.requires_grad)
        is_active = bool(is_trainable and parameter.grad is not None)
        total += count
        if is_trainable:
            trainable += count
            if is_active:
                active += count
            else:
                inactive += count
        gradient_norm = float("nan")
        gradient_nonzero = False
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            gradient_norm = float(gradient.norm().cpu().item())
            gradient_nonzero = bool(torch.count_nonzero(gradient).item())
        rows.append({"name": name, "numel": count, "requires_grad": is_trainable,
                     "graph_active": is_active, "gradient_nonzero": gradient_nonzero,
                     "gradient_norm": gradient_norm})

    model.zero_grad(set_to_none=True)
    model.train(original_training)
    return {"total_params": int(total), "trainable_params": int(trainable),
            "active_params": int(active),
            "inactive_trainable_params": int(inactive),
            "active_ratio": float(active / max(trainable, 1)),
            "parameter_rows": rows}


@torch.inference_mode()
def benchmark_model(model: nn.Module, sample_input: torch.Tensor,
                    device: torch.device, warmup: int = 10, repeats: int = 50,
                    amp_on: bool = False,
                    amp_dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    model.eval()
    warmup = max(1, int(warmup))
    repeats = max(1, int(repeats))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=bool(amp_on)):
            model(sample_input)

    synchronize_device(device)
    start = time.perf_counter()
    for _ in range(repeats):
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=bool(amp_on)):
            model(sample_input)
    synchronize_device(device)
    elapsed = time.perf_counter() - start

    batch_size = int(sample_input.shape[0])
    examples = repeats * batch_size
    peak_memory = float("nan")
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device))

    return {"benchmark_batch_size": batch_size, "benchmark_warmup": warmup,
            "benchmark_repeats": repeats,
            "model_latency_ms_per_location": float(elapsed / max(examples, 1) * 1000.0),
            "model_locations_per_second": float(examples / max(elapsed, 1e-12)),
            "model_peak_memory_bytes": peak_memory}


# =====================================================================
# EMA / state averaging
# =====================================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        state = model.state_dict()
        self.shadow = {k: v.detach().clone().float()
                       for k, v in state.items() if v.dtype.is_floating_point}
        self.other = {k: v.detach().clone()
                      for k, v in state.items() if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, shadow in self.shadow.items():
            shadow.mul_(self.decay).add_(state[key].detach().float(),
                                         alpha=1.0 - self.decay)
        for key in self.other:
            self.other[key].copy_(state[key].detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, shadow in self.shadow.items():
            state[key].copy_(shadow.to(dtype=state[key].dtype,
                                       device=state[key].device))
        for key, value in self.other.items():
            state[key].copy_(value.to(device=state[key].device))


def average_states(states: Sequence[Dict[str, torch.Tensor]]
                   ) -> Optional[Dict[str, torch.Tensor]]:
    states = list(states)
    if not states:
        return None
    if len(states) == 1:
        return {k: v.clone() for k, v in states[0].items()}

    keys = set(states[0].keys())
    if any(set(state.keys()) != keys for state in states[1:]):
        raise ValueError("States have different keys.")

    output: Dict[str, torch.Tensor] = {}
    for key, first in states[0].items():
        if first.dtype.is_floating_point:
            accumulator = torch.zeros_like(first, dtype=torch.float32)
            for state in states:
                accumulator.add_(state[key].float())
            output[key] = (accumulator / float(len(states))).to(first.dtype)
        else:
            output[key] = first.clone()
    return output


# =====================================================================
# Metrics
# =====================================================================

def sigmoid(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60, 60)))


def youden_threshold(labels, probabilities) -> Tuple[float, bool]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(probabilities)
    labels = labels[valid]
    probabilities = probabilities[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5, True
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    objective = tpr - fpr
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5, True
    finite_indices = np.flatnonzero(finite)
    best = finite_indices[int(np.argmax(objective[finite]))]
    unrestricted_best = int(np.argmax(objective))
    repaired = not np.isfinite(thresholds[unrestricted_best])
    return float(thresholds[best]), bool(repaired)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(probabilities)
    labels = labels[valid]
    probabilities = probabilities[valid]

    if len(labels) == 0:
        return {key: float("nan") for key in (
            "auroc", "accuracy", "sensitivity", "precision", "f1", "brier")}

    predictions = (probabilities >= float(threshold)).astype(int)
    has_both = len(np.unique(labels)) == 2

    return {
        "auroc": float(roc_auc_score(labels, probabilities)) if has_both else float("nan"),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)) if has_both else float("nan"),
    }


# =====================================================================
# Training helpers
# =====================================================================

def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
                 smoothing: float, pos_weight: torch.Tensor) -> torch.Tensor:
    if smoothing > 0:
        labels = labels * (1.0 - smoothing) + 0.5 * smoothing
    losses = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none")
    return (losses * weights).mean()


def lr_lambda(epoch: int, total: int, warm: int) -> float:
    epoch = int(epoch)
    total = max(1, int(total))
    warm = max(0, int(warm))
    if warm > 0 and epoch < warm:
        return (epoch + 1) / warm
    progress = (epoch - warm) / max(total - warm, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


# =====================================================================
# Hashing / deterministic seeding
# =====================================================================

def stable_seed(*parts) -> int:
    text = "||".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def seed_torch(value: int) -> None:
    """Re-seed every torch generator cheaply and deterministically."""
    value = int(value) % (2 ** 31 - 1)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(value)
        except Exception:
            pass
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(value)
        except Exception:
            pass


def object_hash(obj) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path, chunk_size=1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stain_columns(available_stains, requested_stains):
    """Map requested stain names onto column indices. Repeats are allowed
    (compute-matched controls reuse one column without duplicating pixels)."""
    available = list(available_stains)
    if len(set(available)) != len(available):
        raise ValueError("Available stain names must be unique.")
    lookup = {name: i for i, name in enumerate(available)}
    missing = [name for name in requested_stains if name not in lookup]
    if missing:
        raise KeyError(f"Missing stains: {missing}")
    return [lookup[name] for name in requested_stains]


# =====================================================================
# Experiment-specification change classification
# =====================================================================

_MISSING = object()

CFG_NEUTRAL_DEFAULTS = {"per_cell_std_floor": PER_CELL_STD_FLOOR_LEGACY}


def classify_config_change(previous: dict, current: dict):
    previous = dict(previous or {})
    current = dict(current or {})
    blocking, notes = [], []

    for key in sorted(set(previous) & set(current)):
        if object_hash(previous[key]) != object_hash(current[key]):
            blocking.append(
                f"config['{key}'] changed: {previous[key]!r} -> {current[key]!r}")

    for key in sorted(set(current) - set(previous)):
        neutral = CFG_NEUTRAL_DEFAULTS.get(key, _MISSING)
        if neutral is _MISSING:
            blocking.append(f"config['{key}'] is new and has no declared neutral default")
        elif object_hash(current[key]) != object_hash(neutral):
            blocking.append(
                f"config['{key}']={current[key]!r} differs from its "
                f"behaviour-neutral default {neutral!r}")
        else:
            notes.append(
                f"config['{key}'] added with its behaviour-neutral default "
                f"{current[key]!r}")

    for key in sorted(set(previous) - set(current)):
        blocking.append(f"config['{key}'] was removed")
    return blocking, notes


# =====================================================================
# Conditioning-aware pipeline comparison
# =====================================================================

UNIT_STD_TOLERANCE = 1e-3
FLAT_RAW_LSB = 1.0


def _plane_correlation(a: torch.Tensor, b: torch.Tensor,
                       chunk: int = 64) -> torch.Tensor:
    """Pearson correlation per row (float64), invariant to per-plane affine
    transforms, so it detects 'wrong stain' rather than last-bit noise."""
    rows = a.shape[0]
    output = torch.empty(rows, dtype=torch.float32)
    for start in range(0, rows, int(chunk)):
        end = min(start + int(chunk), rows)
        aa = a[start:end].double()
        bb = b[start:end].double()
        aa = aa - aa.mean(dim=1, keepdim=True)
        bb = bb - bb.mean(dim=1, keepdim=True)
        norm_a = aa.norm(dim=1)
        norm_b = bb.norm(dim=1)
        numerator = (aa * bb).sum(dim=1)
        both_constant = (norm_a == 0) & (norm_b == 0)
        one_constant = (norm_a == 0) ^ (norm_b == 0)
        value = numerator / (norm_a * norm_b).clamp_min(1e-300)
        value = torch.where(both_constant, torch.ones_like(value), value)
        value = torch.where(one_constant, torch.zeros_like(value), value)
        output[start:end] = value.clamp(-1.0, 1.0).float()
    return output


def plane_agreement(direct: torch.Tensor, cached: torch.Tensor, raw_u8=None,
                    unit_variance_expected: bool = True,
                    unit_std_tol: float = UNIT_STD_TOLERANCE,
                    flat_raw_lsb: float = FLAT_RAW_LSB) -> pd.DataFrame:
    if direct.shape != cached.shape:
        raise ValueError(f"Shape mismatch: {tuple(direct.shape)} vs {tuple(cached.shape)}")
    if direct.ndim != 5:
        raise ValueError("Expected (N, S, 3, H, W) tensors.")

    n, s, c, h, w = direct.shape
    pixels = h * w
    a = direct.detach().to("cpu", torch.float32).reshape(-1, pixels)
    b = cached.detach().to("cpu", torch.float32).reshape(-1, pixels)
    planes = a.shape[0]

    abs_diff = (a - b).abs().amax(dim=1)
    scale = torch.maximum(a.abs().amax(dim=1), b.abs().amax(dim=1)).clamp_min(1.0)
    rel_diff = abs_diff / scale
    a_std = a.std(dim=1, correction=0)
    b_std = b.std(dim=1, correction=0)
    corr = _plane_correlation(a, b)

    if raw_u8 is not None:
        raw = np.asarray(raw_u8)
        if raw.shape != (n, s, h, w, 3):
            raise ValueError(f"raw_u8 shape {raw.shape} does not match {(n, s, h, w, 3)}")
        raw_planes = (np.transpose(raw.astype(np.float32), (0, 1, 4, 2, 3))
                      .reshape(planes, pixels))
        raw_std = raw_planes.std(axis=1)
    else:
        raw_std = np.full(planes, np.nan, dtype=np.float32)

    if unit_variance_expected:
        degenerate = ((a_std.numpy() < 1.0 - float(unit_std_tol))
                      | (b_std.numpy() < 1.0 - float(unit_std_tol)))
    elif raw_u8 is not None:
        degenerate = raw_std <= float(flat_raw_lsb)
    else:
        degenerate = np.zeros(planes, dtype=bool)

    index = np.arange(planes)
    return pd.DataFrame({
        "position": index // (s * c), "stain_slot": (index // c) % s,
        "channel": index % c, "abs_diff": abs_diff.numpy(),
        "scale": scale.numpy(), "rel_diff": rel_diff.numpy(),
        "corr": corr.numpy(), "a_std": a_std.numpy(), "b_std": b_std.numpy(),
        "raw_std": raw_std, "degenerate": degenerate,
    })


def cache_check_verdict(report: pd.DataFrame, rel_tol: float = 2e-3,
                        corr_tol: float = 0.999,
                        flat_raw_lsb: float = FLAT_RAW_LSB) -> dict:
    if not len(report):
        return {"n_planes": 0, "n_checked": 0, "n_degenerate": 0,
                "max_rel_diff": float("nan"), "min_corr": float("nan"),
                "n_failures": 0, "n_suspicious": 0,
                "failures": report, "suspicious": report, "hard_failures": False}

    degenerate = report["degenerate"].to_numpy(bool)
    checked = report.loc[~degenerate]
    if len(checked):
        bad = ((checked["rel_diff"].to_numpy(float) > float(rel_tol))
               | (checked["corr"].to_numpy(float) < float(corr_tol)))
        failures = checked.loc[bad]
        max_rel = float(np.nanmax(checked["rel_diff"].to_numpy(float)))
        min_corr = float(np.nanmin(checked["corr"].to_numpy(float)))
    else:
        failures = checked
        max_rel = float("nan")
        min_corr = float("nan")

    raw_std = report["raw_std"].to_numpy(float)
    suspicious = report.loc[degenerate & (raw_std > float(flat_raw_lsb))]

    return {"n_planes": int(len(report)), "n_checked": int(len(checked)),
            "n_degenerate": int(degenerate.sum()), "max_rel_diff": max_rel,
            "min_corr": min_corr, "n_failures": int(len(failures)),
            "n_suspicious": int(len(suspicious)), "failures": failures,
            "suspicious": suspicious, "hard_failures": bool(len(failures) > 0)}


# =====================================================================
# Model wrappers (input checks + GAP optimisation)
# =====================================================================

class PooledAffineGAP(nn.Module):
    """Algebraic GAP optimisation: mean(affine(x)) == affine(mean(x)).
    Keeps the original GAP head's parameters and state_dict keys."""

    def __init__(self, original):
        super().__init__()
        if original.kind != "gap":
            raise ValueError("PooledAffineGAP only supports the GAP head.")
        self.kind = "gap"
        self.n = original.n
        self.dim = original.dim
        self.projection = original.projection
        self.film_gain = original.film_gain
        self.film_bias = original.film_bias
        self.dropout = original.dropout
        self.classifier = original.classifier

    def forward(self, cells):
        if cells.ndim != 5:
            raise ValueError("Expected cells with shape (B, G, C, H, W).")
        if cells.shape[1] != self.n or cells.shape[2] != self.dim:
            raise ValueError("Unexpected GAP group/channel dimensions.")
        pooled = cells.mean(dim=(-2, -1))
        weight = self.projection.weight[:, :, 0, 0]
        pooled = F.linear(pooled, weight, self.projection.bias)
        pooled = pooled * self.film_gain.unsqueeze(0) + self.film_bias.unsqueeze(0)
        pooled = pooled.mean(dim=1)
        return self.classifier(self.dropout(pooled))


class FusionNet(FusionNetBase):
    """FusionNet with input checks and optional GAP optimisation."""

    def __init__(self, *args, fast_gap=True, **kwargs):
        super().__init__(*args, **kwargs)
        if fast_gap and self.head.kind == "gap":
            if self.enable_aux:
                raise ValueError("fast_gap with auxiliary outputs is not implemented.")
            self.head = PooledAffineGAP(self.head)

    def _pre(self, inputs):
        if inputs.ndim != 5:
            raise RuntimeError(
                f"{self.arm.name}: expected (B, K, 3, H, W), "
                f"received {tuple(inputs.shape)}")
        if inputs.shape[1] != self.k or inputs.shape[2] != 3:
            raise RuntimeError(
                f"{self.arm.name}: expected K={self.k}, C=3; "
                f"received K={inputs.shape[1]}, C={inputs.shape[2]}")
        return super()._pre(inputs)


# =====================================================================
# Resumable EMA
# =====================================================================

class EMA:
    """Resumable EMA over every floating-point state entry."""

    def __init__(self, model, decay=0.999, use_foreach=True):
        self.decay = float(decay)
        self.use_foreach = bool(use_foreach)
        sd = model.state_dict()
        self.floating = {k: v.detach().clone().float()
                         for k, v in sd.items() if v.dtype.is_floating_point}
        self.other = {k: v.detach().clone()
                      for k, v in sd.items() if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        keys = list(self.floating)
        updated = False
        if self.use_foreach:
            try:
                old = [self.floating[k] for k in keys]
                src = [sd[k].detach().float() for k in keys]
                candidate = torch._foreach_mul(old, self.decay)
                torch._foreach_add_(candidate, src, alpha=1.0 - self.decay)
                self.floating = dict(zip(keys, candidate))
                updated = True
            except (RuntimeError, TypeError, AttributeError):
                self.use_foreach = False
                print("[EMA] foreach unavailable; using the scalar path.", flush=True)
        if not updated:
            for k in keys:
                self.floating[k].mul_(self.decay).add_(
                    sd[k].detach().float(), alpha=1.0 - self.decay)
        for k in self.other:
            self.other[k].copy_(sd[k].detach())

    @torch.no_grad()
    def copy_to(self, model):
        target = model.state_dict()
        for k, value in self.floating.items():
            target[k].copy_(value.to(device=target[k].device, dtype=target[k].dtype))
        for k, value in self.other.items():
            target[k].copy_(value.to(device=target[k].device))

    def state_dict(self):
        return {"decay": self.decay, "use_foreach": self.use_foreach,
                "floating": self.floating, "other": self.other}

    def load_state_dict(self, state, device):
        if set(state["floating"]) != set(self.floating):
            raise RuntimeError("EMA floating-state keys do not match.")
        if set(state["other"]) != set(self.other):
            raise RuntimeError("EMA non-floating-state keys do not match.")
        self.decay = float(state["decay"])
        self.use_foreach = bool(state["use_foreach"])
        self.floating = {k: v.to(device=device, dtype=torch.float32).clone()
                         for k, v in state["floating"].items()}
        self.other = {k: v.to(device=device).clone()
                      for k, v in state["other"].items()}


# =====================================================================
# Checkpoint plumbing
# =====================================================================

def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def snapshot_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def capture_rng(device):
    ns = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {"name": ns[0], "keys": torch.tensor(ns[1].astype(np.int64)),
                  "position": int(ns[2]), "has_gauss": int(ns[3]),
                  "cached_gaussian": float(ns[4])},
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "device_type": device.type,
    }
    if device.type == "mps":
        if hasattr(torch.mps, "get_rng_state"):
            try:
                state["torch_mps"] = torch.mps.get_rng_state().cpu().clone()
            except Exception:
                state["torch_mps"] = None
        else:
            state["torch_mps"] = None
    if device.type == "cuda":
        state["torch_cuda"] = [x.cpu().clone()
                               for x in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state, device):
    if state["device_type"] != device.type:
        raise RuntimeError("Resume device type differs from the checkpoint.")
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["name"], ns["keys"].cpu().numpy().astype(np.uint32),
                         int(ns["position"]), int(ns["has_gauss"]),
                         float(ns["cached_gaussian"])))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "mps" and state.get("torch_mps") is not None:
        try:
            torch.mps.set_rng_state(state["torch_mps"].cpu())
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])


def fsync_directory(directory):
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write(path, writer, keep_previous=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    try:
        with open(temporary, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if keep_previous and path.exists():
            os.replace(path, path.with_name(path.name + ".prev"))
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(obj, path):
    content = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True,
                         default=str).encode("utf-8")
    atomic_write(path, lambda handle: handle.write(content))


def atomic_csv(frame, path):
    content = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    atomic_write(path, lambda handle: handle.write(content))


def atomic_checkpoint(payload, path):
    atomic_write(path, lambda handle: torch.save(cpu_tree(payload), handle),
                 keep_previous=True)


def remove_checkpoint(path):
    path = Path(path)
    for candidate in (path, path.with_name(path.name + ".prev"),
                      path.with_name(path.name + ".writing"),
                      path.with_name(path.name + ".unreadable")):
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            pass


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_progress(path, signature):
    path = Path(path)
    previous = path.with_name(path.name + ".prev")
    found = False
    for candidate in (path, previous):
        if not candidate.exists():
            continue
        found = True
        try:
            state = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[CHECKPOINT] Cannot read {candidate.name}: {exc}", flush=True)
            continue
        if state.get("signature") != signature:
            raise RuntimeError("Checkpoint fingerprint mismatch. Do not mix experiments.")
        if candidate == previous:
            print("[CHECKPOINT] Recovered the previous committed snapshot.", flush=True)
            if path.exists():
                os.replace(path, path.with_name(path.name + ".unreadable"))
        return state
    if found:
        raise RuntimeError("Neither the current nor previous checkpoint is readable.")
    return None


# =====================================================================
# Self-tests
# =====================================================================

def self_check() -> None:
    print("train_multiview.py loaded successfully.")
    print(f"Available arms: {len(ARM_LIBRARY)}")
    print(f"Stains: {ALL_STAINS}")
    print(f"Color modes: {COLOR_MODES}")
    print(f"Readout kinds: {CrossStainReadout.KINDS}")
    print(f"2D normalization: {'GroupNorm' if USE_GROUPNORM_2D else 'BatchNorm'}")
    print(f"Per-cell std floor: default={DEFAULT_PER_CELL_STD_FLOOR:g} "
          f"(legacy={PER_CELL_STD_FLOOR_LEGACY:g}, "
          f"one 8-bit level={PER_CELL_STD_FLOOR_ONE_LEVEL:.6f})")
    print(f"stack3d: native Conv3d, kernel_depth={STACK3D_KERNEL_DEPTH}, "
          f"stem_stride={STACK3D_STEM_STRIDE}")

    trunk, dim = build_trunk(DEFAULT_BACKBONE, in_chans=3, pretrained=False)
    print(f"Backbone {DEFAULT_BACKBONE}: out_dim={dim}, "
          f"params={sum(p.numel() for p in trunk.parameters()):,}, "
          f"surviving BatchNorm layers={count_batchnorm(trunk)}")

    x = torch.randn(2, 3, 256, 256)
    trunk.eval()
    with torch.no_grad():
        y = trunk(x)
    print(f"Feature map for a 256px tile: {tuple(y.shape)}")

    x5 = torch.randn(2, 3, 1280, 256)
    with torch.no_grad():
        y5 = trunk(x5)
    print(f"Feature map for a 5-band strip: {tuple(y5.shape)} "
          f"-> cell height {y5.shape[-2] // 5} (must be an integer)")

    blank = np.full((1, 1, 64, 64, 3), 210, dtype=np.uint8)
    for floor in (PER_CELL_STD_FLOOR_LEGACY, PER_CELL_STD_FLOOR_ONE_LEVEL):
        pipe = GPUPipeline(torch.device("cpu"), 64, 64, "plain", True,
                           "stain_norm", per_cell_std_floor=floor)
        with torch.no_grad():
            out = pipe(torch.from_numpy(blank), train=False, aug=AugConfig())
        print(f"Blank plane with floor={floor:g}: output std={float(out.std()):.6f}, "
              f"ULP amplification={1.0 / floor:.3g}")

    for name in ("single_HE", "concat_v5", "concat_v5_cell",
                 "stack3d", "stack3d_HE_TTF1", "ctrl_v_HEx5"):
        spec = ARM_LIBRARY[name]
        print(f"  {name:<24s} mode={spec.mode:<8s} k={spec.k} "
              f"readout={spec.readout:<6s} groups={spec.n_groups()} "
              f"input_elements={input_elements(spec, 256):,}")


def run_self_tests():
    torch.set_num_threads(1)
    torch.manual_seed(19)

    available = ["HE", "TTF1", "MNF116", "p53", "MIB1"]
    token = torch.arange(5).reshape(1, 5, 1, 1, 1)
    for requested in (["p53"], ["MIB1"], ["HE", "TTF1"], available,
                      ["HE"] * 5, ["HE"] * 2):
        cols = stain_columns(available, requested)
        expected = [available.index(x) for x in requested]
        actual = token[:, cols].reshape(-1).tolist()
        assert actual == expected, (requested, actual, expected)
    shuffled = ["p53", "HE", "MIB1", "TTF1", "MNF116"]
    assert stain_columns(shuffled, ["HE", "TTF1"]) == [1, 3]
    print("[TEST] Stain-column mapping passed (including repeated stains).")

    assert CrossStainReadout.KINDS == ("gap", "cell")
    original = CrossStainReadout(kind="gap", n=1, dim=16, dropout=0.0).eval()
    optimized = PooledAffineGAP(copy.deepcopy(original)).eval()
    x1 = torch.randn(2, 1, 16, 4, 5, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = original(x1), optimized(x2)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=2e-6)
    y1.sum().backward()
    y2.sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=2e-6)
    assert set(original.state_dict()) == set(optimized.state_dict())
    print("[TEST] GAP equivalence, gradients and state keys passed.")

    trunk, dim = build_trunk(DEFAULT_BACKBONE, in_chans=3, pretrained=False)
    assert count_batchnorm(trunk) == 0, "BatchNorm survived the conversion."
    print(f"[TEST] 2D trunk contains no BatchNorm (out_dim={dim}).")

    with torch.no_grad():
        feature = trunk.eval()(torch.randn(1, 3, 1280, 256))
    assert feature.shape[-2] % 5 == 0, "Strip feature map does not divide by 5."
    print(f"[TEST] 5-band strip feature map {tuple(feature.shape)} splits "
          f"exactly into 5 cells of height {feature.shape[-2] // 5}.")

    for name in ("concat_v5_cell", "concat_v2_HE_TTF1_cell", "ctrl_v_HEx5"):
        spec = ARM_LIBRARY[name]
        model = FusionNet(arm=spec, backbone=DEFAULT_BACKBONE, pretrained=False,
                          dropout=0.0, stain_code=True, fast_gap=True).eval()
        with torch.no_grad():
            out = model(torch.randn(2, spec.k, 3, 128, 128))
        assert out.shape == (2, 1), (name, out.shape)
        del model
    print("[TEST] Cell-readout and control arms build and run.")

    metrics = binary_metrics([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.9], 0.5)
    assert set(metrics) == {"auroc", "accuracy", "sensitivity",
                            "precision", "f1", "brier"}, set(metrics)
    print(f"[TEST] binary_metrics returns exactly {sorted(metrics)}.")

    thr, repaired = youden_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert np.isfinite(thr) and not repaired
    print(f"[TEST] Youden threshold selection passed (threshold={thr:.3f}).")

    rng = np.random.default_rng(7)
    raw = rng.integers(0, 255, size=(2, 3, 32, 32, 3), dtype=np.uint8)
    raw[:, 1] = 203
    pipe = GPUPipeline(torch.device("cpu"), 32, 32, "plain", True,
                       "stain_norm", per_cell_std_floor=PER_CELL_STD_FLOOR_LEGACY)
    with torch.no_grad():
        out = pipe(torch.from_numpy(raw), train=False, aug=AugConfig())

    same = plane_agreement(out, out.clone(), raw_u8=raw)
    assert len(same) == 2 * 3 * 3
    assert float(same["rel_diff"].max()) == 0.0
    assert int(same["degenerate"].sum()) == 2 * 3, same["degenerate"].sum()
    verdict = cache_check_verdict(same)
    assert not verdict["hard_failures"]
    assert verdict["n_degenerate"] == 6 and verdict["n_checked"] == 12
    print("[TEST] plane_agreement flags blank planes and passes identity.")

    swapped = out.clone()
    swapped[:, [0, 2]] = out[:, [2, 0]]
    report = plane_agreement(out, swapped, raw_u8=raw)
    bad = cache_check_verdict(report)
    assert bad["hard_failures"], "A stain swap must be detected."
    assert bad["min_corr"] < 0.5
    print("[TEST] A swapped stain column is detected by the correlation gate.")

    perturbed = out.clone()
    perturbed[:, 1] += 0.05
    assert not cache_check_verdict(
        plane_agreement(out, perturbed, raw_u8=raw))["hard_failures"]
    perturbed = out.clone()
    perturbed[:, 0] += 0.05
    assert cache_check_verdict(
        plane_agreement(out, perturbed, raw_u8=raw))["hard_failures"]
    print("[TEST] Degenerate planes excluded, healthy planes still gated.")

    base = {"epochs": 20, "lr": 2e-4}
    blocking, notes = classify_config_change(base, dict(base))
    assert not blocking and not notes
    blocking, notes = classify_config_change(
        base, {**base, "per_cell_std_floor": PER_CELL_STD_FLOOR_LEGACY})
    assert not blocking and len(notes) == 1, (blocking, notes)
    blocking, _ = classify_config_change(
        base, {**base, "per_cell_std_floor": 1.0 / 255.0})
    assert blocking, "A non-neutral new key must block."
    blocking, _ = classify_config_change(base, {**base, "epochs": 32})
    assert blocking, "Changing a shared key must block."
    blocking, _ = classify_config_change(base, {"epochs": 20})
    assert blocking, "Removing a key must block."
    print("[TEST] Config-change classification passed.")

    device = torch.device("cpu")
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    targets = (torch.arange(8) % 2).float().reshape(8, 1)

    def construct():
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(8, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 0.95 ** e)
        ema = EMA(model, decay=0.9, use_foreach=False)
        return model, optimizer, scheduler, ema

    def step(model, optimizer, scheduler, ema, tag):
        seed_torch(stable_seed("unit-step", tag))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(features), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)

    set_seed(123, strict=True)
    model, optimizer, scheduler, ema = construct()
    step(model, optimizer, scheduler, ema, "a")
    step(model, optimizer, scheduler, ema, "b")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.pt"
        atomic_checkpoint({"signature": "self-test", "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(),
                           "ema": ema.state_dict(),
                           "rng": capture_rng(device)}, path)

        step(model, optimizer, scheduler, ema, "c")
        expected_model = snapshot_state(model)
        expected_ema = {k: v.clone() for k, v in ema.floating.items()}

        set_seed(999, strict=True)
        model2, optimizer2, scheduler2, ema2 = construct()
        saved = load_progress(path, "self-test")
        model2.load_state_dict(saved["model"])
        scheduler2.load_state_dict(saved["scheduler"])
        optimizer2.load_state_dict(saved["optimizer"])
        ema2.load_state_dict(saved["ema"], device)
        restore_rng(saved["rng"], device)
        step(model2, optimizer2, scheduler2, ema2, "c")

        for k, value in expected_model.items():
            torch.testing.assert_close(value, model2.state_dict()[k], rtol=0, atol=0)
        for k, value in expected_ema.items():
            torch.testing.assert_close(value, ema2.floating[k], rtol=0, atol=0)

    print("[TEST] Optimizer/scheduler/EMA resume check passed.")
    print("These are unit tests, not an end-to-end test of your tile bank.")


if __name__ == "__main__":
    self_check()
    print()
    run_self_tests()

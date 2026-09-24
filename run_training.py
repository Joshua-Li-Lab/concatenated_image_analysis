"""
run_training_colab.py

Resumable slide-level leave-one-group-out multi-stain fusion trainer.
"""

from __future__ import annotations
import os
import argparse
import contextlib
import copy
import fcntl
import gc
import importlib.metadata
import inspect as _inspect
import math
import platform
import shutil
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

import train_multiview as S

T = S

try:
    import report_excel as REPORT
except Exception as _exc:
    REPORT = None
    _REPORT_ERROR = _exc


LEGACY_PER_CELL_STD_FLOOR = float(getattr(T, "PER_CELL_STD_FLOOR_LEGACY", 1e-5))
ONE_LEVEL_PER_CELL_STD_FLOOR = 1.0 / 255.0


CFG = {
    "split_level": "slide",

    "arms": [
        "single_HE", "single_TTF1", "single_MNF116", "single_p53", "single_MIB1",
        "concat_v2_HE_TTF1", "concat_v2_HE_MNF116", "concat_v2_HE_p53", "concat_v2_HE_MIB1",
        "concat_v5", "concat_v5_cell",
        "stack3d", "stack3d_HE_TTF1",
        "ctrl_v_HEx5", "concat_v2_HE_TTF1_cell",
    ],
    "arm_overrides": {},

    "seeds": [0, 1, 2],

    "backbone": "timm:resnet10t",
    "pretrained": False,
    "tile_size": 256,
    "batch_size": 64,
    "arm_batch_size_overrides": {},

    "dropout": 0.3,
    "stain_code": True,
    "cell_gutter": 0,
    "mixer_layers": 2,
    "readout_rank": 32,
    "readout_heads": 4,
    "readout_vrank": 64,
    "stack3d_base_range": [8, 160, 2],
    "fast_gap": True,

    "epochs": 20,
    "min_epochs": 5,
    "patience": 5,
    "min_delta": 1e-5,
    "val_smooth": 3,
    "swa_top_k": 4,

    "lr": 2e-4,
    "lr_ref_batch": 32,
    "scale_lr_with_batch": True,
    "head_lr_mult": 5.0,
    "weight_decay": 3e-4,
    "warmup_epochs": 2,
    "label_smoothing": 0.05,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "fast_ema": True,

    "color_mode": "stain_norm",
    "per_cell_standardize": True,
    "per_cell_std_floor": ONE_LEVEL_PER_CELL_STD_FLOOR,

    "augment": True,
    "aug": {
        "dihedral": True, "jitter": 0.20, "jitter_p": 0.9,
        "hed_sigma": 0.20, "hed_bias": 0.08, "hed_p": 0.95,
        "gray_p": 0.15, "stain_dropout": 0.0,
        "cutout_p": 0.0, "cutout_frac": 0.0,
    },

    "tta_epoch": 1,
    "tta_final": 2,

    "use_selected_tiles_only": True,
    "border_filter_mode": "um",
    "signed_border_dist_um": 60.0,
    "signed_border_dist_px": 300.0,

    "split_tile_budgets": {"train": 6000, "validation": 1000, "test": None},
    "max_tiles_per_fold": 6000,
    "fold_tile_ratios": {"train": 0.62, "validation": 0.14, "test": 0.24},

    "shared_full_stain_batch": True,

    "use_amp": True,
    "strict_determinism": False,
    "eval_batch_size": 128,
    "global_seed": 42,

    "co_train_group_size": 16,
    "prefetch_depth": 4,

    "cache_eval_preprocess": True,
    "eval_cache_dtype": "float32",
    "eval_cache_build_batch": 16,
    "validate_eval_cache": True,
    "delete_eval_cache_after_fold": True,
    "eval_cache_plane_stats": True,

    "val_cache_on_gpu": "auto",
    "val_cache_gpu_max_fraction": 0.25,

    "cache_training_tiles": True,
    "train_cache_max_gb": 16.0,
    "train_cache_available_fraction": 0.45,

    "benchmark_repeats": 5,
    "delete_group_checkpoint_when_done": True,

    "report_baseline_arm": "single_HE",
    "write_excel_report": True,
}


RECIPE_KEYS = (
    "split_level", "backbone", "pretrained", "tile_size", "batch_size",
    "dropout", "stain_code", "cell_gutter", "mixer_layers",
    "readout_rank", "readout_heads", "readout_vrank",
    "stack3d_base_range", "fast_gap",
    "epochs", "min_epochs", "patience", "min_delta", "val_smooth",
    "swa_top_k", "lr", "lr_ref_batch", "scale_lr_with_batch",
    "head_lr_mult", "weight_decay", "warmup_epochs", "label_smoothing",
    "grad_clip", "ema_decay", "fast_ema",
    "color_mode", "per_cell_standardize", "per_cell_std_floor",
    "augment", "aug",
    "tta_epoch", "tta_final",
    "use_selected_tiles_only", "border_filter_mode",
    "signed_border_dist_um", "signed_border_dist_px",
    "split_tile_budgets", "max_tiles_per_fold", "fold_tile_ratios",
    "shared_full_stain_batch",
    "use_amp", "strict_determinism", "eval_batch_size", "global_seed",
)

RECIPE_NEUTRAL_DEFAULTS = {
    "per_cell_std_floor": LEGACY_PER_CELL_STD_FLOOR,
    "split_tile_budgets": None,
}

ARM_RECIPE_KEYS = (
    "dropout", "stain_code", "cell_gutter", "mixer_layers",
    "readout_rank", "readout_heads", "readout_vrank",
    "stack3d_base_range", "fast_gap",
)

ENVIRONMENT_SPEC_KEYS = ("device_type", "python", "versions")
PROVENANCE_SPEC_KEYS = ("support_version", "source_sha256")

CACHE_CHECK_MODES = ("strict", "report", "off")
UNIT_STD_TOLERANCE = 1e-3
FLAT_RAW_LSB = 1.0

CHECK_POLICY = {
    "builder_replay": {"rel_tol": 1e-3, "corr_tol": 0.999,
                       "degenerate_rel_tol": 0.05},
    "tta_equivariance": {"rel_tol": 2e-3, "corr_tol": 0.999,
                         "degenerate_rel_tol": None},
    "stain_selection": {"rel_tol": 2e-3, "corr_tol": 0.999,
                        "degenerate_rel_tol": None},
}

MIRROR_SKIP_TOP = {"eval_cache", "progress"}


class PauseRequested(Exception):
    pass


class StopController:
    def __init__(self, output_dir, pause_after_batches=0, max_hours=0.0):
        self.path = Path(output_dir) / "PAUSE"
        self.requested = False
        self.signal_count = 0
        self.batch_count = 0
        self.pause_after_batches = int(pause_after_batches)
        self.start = time.monotonic()
        self.limit = float(max_hours) * 3600.0 if max_hours else 0.0
        self._announced = False

        for name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), self._handler)

    def _handler(self, signum, frame):
        self.signal_count += 1
        if self.signal_count >= 2:
            os.write(2, b"\nForced exit. Resume uses the last checkpoint.\n")
            os._exit(130)
        self.requested = True
        os.write(2, b"\nPause requested. Waiting for a safe boundary...\n")

    def elapsed_hours(self):
        return (time.monotonic() - self.start) / 3600.0

    def time_left(self):
        if not self.limit:
            return float("inf")
        return self.limit - (time.monotonic() - self.start)

    def after_batch(self):
        self.batch_count += 1
        if (self.pause_after_batches > 0
                and self.batch_count >= self.pause_after_batches):
            self.requested = True

    def check(self):
        if self.limit and self.time_left() <= 0 and not self.requested:
            self.requested = True
            if not self._announced:
                self._announced = True
                print("[TIME LIMIT] Reached --max-hours; saving at a safe "
                      "point and exiting.", flush=True)
        if self.requested or self.path.exists():
            self.requested = True
            raise PauseRequested()

    def acknowledge(self):
        if self.path.exists():
            self.path.unlink()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fmt(value):
    try:
        value = float(value)
        return f"{value:.4f}" if np.isfinite(value) else "NaN"
    except (TypeError, ValueError):
        return "NaN"


def configure_backends(device):
    """Speed settings. TF32 stays off to protect fp32 HED colour-deconvolution
    precision."""
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {name}  VRAM={total:.1f}GB", flush=True)


def gpu_name():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def release_device(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def memory_message(device):
    vm = psutil.virtual_memory()
    message = f"RAM available={vm.available / 1e9:.1f}GB"
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        message += f" | VRAM free={free/1e9:.1f}/{total/1e9:.1f}GB"
    elif device.type == "mps":
        message += (
            f" | MPS current={torch.mps.current_allocated_memory()/1e9:.2f}GB"
        )
    return message


def group_key_of(slide, split_level):
    slide = str(slide)
    if split_level == "slide":
        return slide
    raise ValueError(f"Unknown split_level: {split_level} (slide-level only).")


def batch_size_for(cfg, arm_name):
    return int(cfg["arm_batch_size_overrides"].get(arm_name, cfg["batch_size"]))


def base_model_kwargs(cfg):
    return {
        "dropout": float(cfg["dropout"]),
        "stain_code": bool(cfg["stain_code"]),
        "cell_gutter": int(cfg["cell_gutter"]),
        "mixer_layers": int(cfg["mixer_layers"]),
        "readout_rank": int(cfg["readout_rank"]),
        "readout_heads": int(cfg["readout_heads"]),
        "readout_vrank": int(cfg["readout_vrank"]),
        "stack3d_base_range": tuple(cfg["stack3d_base_range"]),
        "enable_aux": False,
        "fast_gap": bool(cfg["fast_gap"]),
    }


def arm_model_kwargs(cfg, arm_name):
    kwargs = base_model_kwargs(cfg)
    for key, value in cfg.get("arm_overrides", {}).get(arm_name, {}).items():
        if key not in ARM_RECIPE_KEYS:
            raise KeyError(
                f"arm_overrides['{arm_name}']: '{key}' cannot be overridden. "
                f"Allowed: {sorted(ARM_RECIPE_KEYS)}"
            )
        if key == "stack3d_base_range":
            value = tuple(value)
        kwargs[key] = value
    return kwargs


def arm_recipe(cfg, arm_name):
    spec = T.ARM_LIBRARY[arm_name]
    kwargs = arm_model_kwargs(cfg, arm_name)
    return {
        "arm": arm_name,
        "mode": spec.mode,
        "stains": list(spec.stains),
        "readout": spec.readout,
        "n_groups": spec.n_groups(),
        "batch_size": batch_size_for(cfg, arm_name),
        "model_kwargs": {k: kwargs[k] for k in sorted(kwargs)},
    }


def arm_signature(cfg, arm_name):
    return S.object_hash(arm_recipe(cfg, arm_name))


def make_model(cfg, arm_name, device):
    return S.FusionNet(
        arm=T.ARM_LIBRARY[arm_name],
        backbone=cfg["backbone"],
        pretrained=cfg["pretrained"],
        **arm_model_kwargs(cfg, arm_name),
    ).to(device)


def make_benchmark_model(cfg, arm_name, device, state):
    kwargs = dict(arm_model_kwargs(cfg, arm_name))
    kwargs["fast_gap"] = False
    model = S.FusionNet(
        arm=T.ARM_LIBRARY[arm_name],
        backbone=cfg["backbone"],
        pretrained=cfg["pretrained"],
        **kwargs,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def pipeline_accepts_std_floor():
    try:
        parameters = _inspect.signature(T.GPUPipeline.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "per_cell_std_floor" in parameters


def configured_std_floor(cfg):
    return float(cfg.get("per_cell_std_floor", LEGACY_PER_CELL_STD_FLOOR))


def make_pipe(bank, cfg, device):
    kwargs = dict(
        device=device,
        tile_size=cfg["tile_size"],
        bank_size=bank.size,
        normalise="imagenet" if cfg["pretrained"] else "plain",
        per_cell_standardize=cfg["per_cell_standardize"],
        color_mode=cfg["color_mode"],
    )
    floor = configured_std_floor(cfg)
    if pipeline_accepts_std_floor():
        kwargs["per_cell_std_floor"] = floor
    elif floor != LEGACY_PER_CELL_STD_FLOOR:
        raise RuntimeError(
            f"Requested per_cell_std_floor={floor!r}, but the installed "
            f"{Path(T.__file__).name} hard-codes {LEGACY_PER_CELL_STD_FLOOR!r}."
        )
    return T.GPUPipeline(**kwargs)


def effective_std_floor(pipe, cfg):
    return float(getattr(pipe, "per_cell_std_floor", configured_std_floor(cfg)))


def checked_metrics(labels, probabilities, threshold):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)

    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("Invalid prediction/label dimensions.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite prediction probabilities.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities are outside [0, 1].")
    if not np.isfinite(threshold):
        raise ValueError("Threshold is not finite.")

    result = T.binary_metrics(y, p, threshold)
    result["brier"] = float(np.mean((p - y) ** 2))
    if not np.any(y == 1):
        result["sensitivity"] = float("nan")
    return result


def _plane_correlation(a, b, chunk=64):
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


def plane_agreement(direct, cached, raw_u8=None, unit_variance_expected=True,
                    unit_std_tol=UNIT_STD_TOLERANCE, flat_raw_lsb=FLAT_RAW_LSB):
    if direct.shape != cached.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(direct.shape)} vs {tuple(cached.shape)}"
        )
    if direct.ndim != 5:
        raise ValueError("Expected (N, S, 3, H, W) tensors.")

    n, s, c, h, w = direct.shape
    pixels = h * w
    a = direct.detach().to("cpu", torch.float32).contiguous().reshape(-1, pixels)
    b = cached.detach().to("cpu", torch.float32).contiguous().reshape(-1, pixels)
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
            raise ValueError(
                f"raw_u8 shape {raw.shape} does not match {(n, s, h, w, 3)}"
            )
        raw_planes = np.ascontiguousarray(
            np.transpose(raw.astype(np.float32), (0, 1, 4, 2, 3))
        ).reshape(planes, pixels)
        raw_std = raw_planes.std(axis=1)
        del raw_planes
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
        "position": index // (s * c),
        "stain_slot": (index // c) % s,
        "channel": index % c,
        "abs_diff": abs_diff.numpy(),
        "scale": scale.numpy(),
        "rel_diff": rel_diff.numpy(),
        "corr": corr.numpy(),
        "a_std": a_std.numpy(),
        "b_std": b_std.numpy(),
        "raw_std": raw_std,
        "degenerate": degenerate,
    })


def cache_check_verdict(report, rel_tol, corr_tol, degenerate_rel_tol=None,
                        flat_raw_lsb=FLAT_RAW_LSB):
    empty = report.iloc[0:0]
    if not len(report):
        return {"n_planes": 0, "n_checked": 0, "n_degenerate": 0,
                "max_rel_diff": float("nan"), "min_corr": float("nan"),
                "n_failures": 0, "n_suspicious": 0,
                "failures": empty, "suspicious": empty, "hard_failures": False}

    degenerate = report["degenerate"].to_numpy(bool)
    checked = report.loc[~degenerate]
    flagged = [empty]

    if len(checked):
        bad = ((checked["rel_diff"].to_numpy(float) > float(rel_tol))
               | (checked["corr"].to_numpy(float) < float(corr_tol)))
        flagged.append(checked.loc[bad])
        max_rel = float(np.nanmax(checked["rel_diff"].to_numpy(float)))
        min_corr = float(np.nanmin(checked["corr"].to_numpy(float)))
    else:
        max_rel = float("nan")
        min_corr = float("nan")

    if degenerate_rel_tol is not None:
        rough = report.loc[degenerate]
        if len(rough):
            structural = (rough["rel_diff"].to_numpy(float)
                          > float(degenerate_rel_tol))
            flagged.append(rough.loc[structural])

    failures = pd.concat(flagged, ignore_index=False)
    raw_std = report["raw_std"].to_numpy(float)
    suspicious = report.loc[degenerate & (raw_std > float(flat_raw_lsb))]

    return {
        "n_planes": int(len(report)), "n_checked": int(len(checked)),
        "n_degenerate": int(degenerate.sum()),
        "max_rel_diff": max_rel, "min_corr": min_corr,
        "n_failures": int(len(failures)), "n_suspicious": int(len(suspicious)),
        "failures": failures, "suspicious": suspicious,
        "hard_failures": bool(len(failures) > 0),
    }


def resolve_budgets(cfg):
    budgets = cfg.get("split_tile_budgets")
    if budgets is None:
        order = ["train", "validation", "test"]
        exact = np.array(
            [cfg["max_tiles_per_fold"] * cfg["fold_tile_ratios"][k] for k in order],
            dtype=float,
        )
        integers = np.floor(exact).astype(int)
        remainder = int(cfg["max_tiles_per_fold"] - integers.sum())
        for j in np.argsort(-(exact - integers), kind="stable")[:remainder]:
            integers[j] += 1
        return dict(zip(order, [int(v) for v in integers]))

    out = {}
    for key in ("train", "validation", "test"):
        value = budgets.get(key, None)
        out[key] = None if value in (None, 0, "all", "ALL") else int(value)
    return out


def validate_cfg(cfg):
    if cfg["split_level"] != "slide":
        raise ValueError("split_level must be 'slide' (patient-level splitting "
                         "has been removed from this build).")
    if not cfg["arms"]:
        raise ValueError("No arm selected.")
    if len(set(cfg["arms"])) != len(cfg["arms"]):
        raise ValueError("cfg['arms'] has duplicates.")
    for arm in cfg["arms"]:
        if arm not in T.ARM_LIBRARY:
            raise KeyError(f"Unknown arm: {arm}. Available: {sorted(T.ARM_LIBRARY)}")
        if batch_size_for(cfg, arm) <= 0:
            raise ValueError("batch size must be positive.")
        arm_model_kwargs(cfg, arm)
    for arm in cfg.get("arm_overrides", {}):
        if arm not in T.ARM_LIBRARY:
            raise KeyError(f"arm_overrides points to unknown arm: {arm}")

    if len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds has duplicates.")
    if not cfg["seeds"] or any(int(s) < 0 for s in cfg["seeds"]):
        raise ValueError("seeds must be non-negative integers.")

    if not 1 <= cfg["min_epochs"] <= cfg["epochs"]:
        raise ValueError("Require 1 <= min_epochs <= epochs.")
    if cfg["warmup_epochs"] >= cfg["epochs"]:
        raise ValueError("warmup_epochs must be less than epochs.")
    if cfg["swa_top_k"] < 1 or cfg["val_smooth"] < 1:
        raise ValueError("swa_top_k / val_smooth must be positive.")
    if cfg["co_train_group_size"] < 1:
        raise ValueError("co_train_group_size must be positive.")

    for key in ("tta_epoch", "tta_final"):
        if not 1 <= int(cfg[key]) <= len(T.TTA_VIEWS):
            raise ValueError(f"{key} must be in 1..{len(T.TTA_VIEWS)}")

    ratios = cfg["fold_tile_ratios"]
    if set(ratios) != {"train", "validation", "test"}:
        raise ValueError("fold_tile_ratios needs train/validation/test.")
    if any(float(v) <= 0 for v in ratios.values()):
        raise ValueError("Ratios must be positive.")
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Ratios must sum to 1.")

    budgets = resolve_budgets(cfg)
    if budgets["train"] is not None and budgets["train"] < 200:
        raise ValueError("Training tile budget is too small.")
    if budgets["validation"] is not None and budgets["validation"] < 50:
        raise ValueError("Validation tile budget is too small.")

    if cfg["border_filter_mode"] not in ("um", "px", "none"):
        raise ValueError("border_filter_mode must be 'um'/'px'/'none'.")
    if cfg["eval_cache_dtype"] != "float32":
        raise ValueError("This version only supports a float32 eval cache.")

    floor = configured_std_floor(cfg)
    if not floor > 0.0:
        raise ValueError("per_cell_std_floor must be positive.")
    if floor > 0.5:
        raise ValueError("per_cell_std_floor is too large.")
    if int(cfg["eval_cache_build_batch"]) <= 0:
        raise ValueError("eval_cache_build_batch must be positive.")
    if cfg["val_cache_on_gpu"] not in ("auto", "on", "off"):
        raise ValueError("val_cache_on_gpu must be auto/on/off.")


def dataset_fingerprint(bank):
    """No mtime. Re-copying the data must not force a full retrain."""
    root = Path(bank.bank_dir)
    shards = []
    for slide, entry in sorted(bank.meta["shards"].items()):
        path = root / entry["file"]
        shards.append({
            "slide": str(slide),
            "file": str(entry["file"]),
            "size": int(path.stat().st_size),
        })
    return {
        "bank_meta_sha256": S.file_hash(root / "bank_meta.json"),
        "bank_index_sha256": S.file_hash(root / "bank_index.csv"),
        "shards": shards,
        "shard_check": "file size + bank_meta/bank_index content hashes",
    }


def dataset_detail(bank):
    root = Path(bank.bank_dir)
    rows = []
    for slide, entry in sorted(bank.meta["shards"].items()):
        stat = (root / entry["file"]).stat()
        rows.append({"slide": str(slide), "file": str(entry["file"]),
                     "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return {"recorded_at": utc_now(), "bank_dir": str(root), "shards": rows}


def software_fingerprint():
    versions = {}
    for name in ("torch", "torchvision", "timm", "numpy",
                 "pandas", "scikit-learn", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def recipe_specification(cfg, bank, device):
    return {
        "support_version": S.VERSION,
        "recipe": {key: cfg[key] for key in RECIPE_KEYS},
        "groupnorm_2d": bool(T.USE_GROUPNORM_2D),
        "stack3d_kernel_depth": int(T.STACK3D_KERNEL_DEPTH),
        "stack3d_stem_stride": int(T.STACK3D_STEM_STRIDE),
        "device_type": device.type,
        "python": platform.python_version(),
        "versions": software_fingerprint(),
        "source_sha256": {
            Path(__file__).name: S.file_hash(__file__),
            Path(T.__file__).name: S.file_hash(T.__file__),
        },
        "dataset": dataset_fingerprint(bank),
    }


_MISSING = object()


def _same(a, b):
    return S.object_hash(a) == S.object_hash(b)


def classify_mapping_change(previous, current, neutral_defaults, label):
    previous = dict(previous or {})
    current = dict(current or {})
    blocking, notes = [], []

    for key in sorted(set(previous) & set(current)):
        if not _same(previous[key], current[key]):
            blocking.append(f"{label}['{key}'] changed: "
                            f"{previous[key]!r} -> {current[key]!r}")
    for key in sorted(set(current) - set(previous)):
        neutral = (neutral_defaults or {}).get(key, _MISSING)
        if neutral is _MISSING:
            blocking.append(f"{label}['{key}'] is new and has no declared "
                            "neutral default")
        elif not _same(current[key], neutral):
            blocking.append(f"{label}['{key}']={current[key]!r} differs from "
                            f"its neutral default {neutral!r}")
        else:
            notes.append(f"{label}['{key}'] added with its neutral default "
                         f"{current[key]!r}")
    for key in sorted(set(previous) - set(current)):
        blocking.append(f"{label}['{key}'] was removed")
    return blocking, notes


def classify_dataset_change(previous, current):
    if _same(previous, current):
        return [], []
    previous = dict(previous or {})
    current = dict(current or {})
    blocking = []
    for key in ("bank_meta_sha256", "bank_index_sha256"):
        if previous.get(key) != current.get(key):
            blocking.append(f"dataset['{key}'] changed: bank content is not "
                            "the original")

    def identity(entries):
        return {(str(e.get("slide")), str(e.get("file"))): int(e.get("size", -1))
                for e in (entries or [])}

    if identity(previous.get("shards")) != identity(current.get("shards")):
        blocking.append("dataset['shards'] slide/file names or byte sizes "
                        "changed")
    return blocking, []


def classify_specification_change(previous, current, allow_environment_change):
    previous = dict(previous or {})
    current = dict(current or {})
    blocking, notes = [], []

    recipe_blocking, recipe_notes = classify_mapping_change(
        previous.get("recipe"), current.get("recipe"),
        RECIPE_NEUTRAL_DEFAULTS, "recipe",
    )
    blocking.extend(recipe_blocking)
    notes.extend(recipe_notes)

    for key in ("groupnorm_2d", "stack3d_kernel_depth", "stack3d_stem_stride"):
        if not _same(previous.get(key), current.get(key)):
            blocking.append(f"'{key}' changed: model architecture differs")

    dataset_blocking, dataset_notes = classify_dataset_change(
        previous.get("dataset"), current.get("dataset")
    )
    blocking.extend(dataset_blocking)
    notes.extend(dataset_notes)

    for key in ENVIRONMENT_SPEC_KEYS:
        if _same(previous.get(key), current.get(key)):
            continue
        message = f"'{key}' changed: {previous.get(key)!r} -> {current.get(key)!r}"
        if allow_environment_change:
            notes.append(message + " (accepted via --allow-environment-change)")
        else:
            blocking.append(message + "; this changes floating-point results. "
                            "If acceptable, pass --allow-environment-change and "
                            "document it in the paper.")

    for key in PROVENANCE_SPEC_KEYS:
        if _same(previous.get(key), current.get(key)):
            continue
        if key == "source_sha256":
            old = dict(previous.get(key) or {})
            new = dict(current.get(key) or {})
            changed = sorted(n for n in set(old) | set(new)
                             if old.get(n) != new.get(n))
            notes.append("source_sha256 changed: " + ", ".join(changed))
        else:
            notes.append(f"'{key}' changed: {previous.get(key)!r} -> "
                         f"{current.get(key)!r}")
    return blocking, notes


@contextlib.contextmanager
def exclusive_lock(path, message):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(message) from exc
        except OSError as exc:
            print(f"[LOCK] This filesystem does not support file locks ({exc}); "
                  "continuing, but make sure only one process writes this "
                  "directory.", flush=True)
        try:
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        handle.close()


def count_completed_markers(directory):
    runs = Path(directory) / "runs"
    if not runs.is_dir():
        return 0
    return sum(1 for _ in runs.glob("*/complete.json"))


def record_amendment(marker_path, stored_signature, fresh_signature,
                     specification, notes):
    document = S.read_json(marker_path)
    amendments = list(document.get("amendments", []))
    if any(str(e.get("amended_specification_signature")) == fresh_signature
           for e in amendments):
        return len(amendments)
    amendments.append({
        "recorded_at": utc_now(),
        "amended_specification_signature": fresh_signature,
        "notes": list(notes),
        "specification": specification,
    })
    document["amendments"] = amendments
    document["signature"] = stored_signature
    S.atomic_json(document, marker_path)
    return len(amendments)


def resolve_output_dir(base, specification, cfg, args):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    fresh_signature = S.object_hash(specification)
    allow_environment = bool(args.allow_environment_change)

    with exclusive_lock(base / ".resolve.lock",
                        "Another process is resolving this base directory."):
        forced = args.resume_dir or args.run_name
        if forced:
            candidate = Path(forced).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            if not (candidate / "experiment.json").exists():
                if args.resume_dir:
                    raise RuntimeError(
                        f"--resume-dir has no experiment.json: {candidate}")
                candidate.mkdir(parents=True, exist_ok=True)
                S.atomic_json(
                    {"signature": fresh_signature, "created_at": utc_now(),
                     "specification": specification, "amendments": []},
                    candidate / "experiment.json",
                )
                print(f"[DIR] Created {candidate.name}", flush=True)
                S.atomic_json(cfg, candidate / "config_used.json")
                return candidate, fresh_signature, {"mode": "created", "notes": []}
            candidates = [candidate.resolve()]
        else:
            candidates = sorted(p for p in base.iterdir() if p.is_dir())

        compatible, rejected = [], []
        for candidate in candidates:
            marker = candidate / "experiment.json"
            if not marker.exists():
                continue
            try:
                document = S.read_json(marker)
            except Exception as exc:
                rejected.append(
                    (candidate, [f"experiment.json unreadable: {exc}"]))
                continue

            stored_signature = str(document.get("signature", ""))
            if stored_signature == fresh_signature:
                print(f"[DIR] Reusing {candidate.name}", flush=True)
                S.atomic_json(cfg, candidate / "config_used.json")
                return candidate, fresh_signature, {"mode": "exact", "notes": []}

            blocking, notes = classify_specification_change(
                document.get("specification", {}), specification, allow_environment
            )
            if blocking:
                rejected.append((candidate, blocking))
            else:
                compatible.append((candidate, stored_signature, notes,
                                   count_completed_markers(candidate)))

        if compatible:
            compatible.sort(key=lambda item: (item[3], item[0].name), reverse=True)
            directory, stored_signature, notes, done = compatible[0]
            total = record_amendment(directory / "experiment.json", stored_signature,
                                     fresh_signature, specification, notes)
            print(f"[DIR] Adopting {directory.name} (keeping original "
                  "signature)", flush=True)
            for note in notes:
                print(f"      - {note}", flush=True)
            print(f"      Kept {done} completed run(s); experiment.json now "
                  f"records {total} amendment(s).", flush=True)
            S.atomic_json(cfg, directory / "config_used.json")
            return directory, stored_signature, {"mode": "adopted", "notes": notes}

        if forced and rejected:
            candidate, reasons = rejected[0]
            raise RuntimeError(
                f"{candidate.name} cannot be reused:\n  - "
                + "\n  - ".join(reasons)
                + "\n\nThese differences change training semantics. Use a "
                "different --run-name to start a new directory."
            )

        if rejected:
            print("[DIR] No compatible directory. Reasons:", flush=True)
            for candidate, reasons in rejected:
                print(f"  {candidate.name}", flush=True)
                for reason in reasons[:6]:
                    print(f"    - {reason}", flush=True)
                if len(reasons) > 6:
                    print(f"    - ... {len(reasons) - 6} more", flush=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{stamp}__{cfg['split_level']}__{fresh_signature[:10]}"
        directory = base / name
        directory.mkdir(parents=True, exist_ok=False)
        S.atomic_json(
            {"signature": fresh_signature, "created_at": utc_now(),
             "specification": specification, "amendments": []},
            directory / "experiment.json",
        )
        print(f"[DIR] Created {directory.name}", flush=True)
        S.atomic_json(cfg, directory / "config_used.json")
        return directory, fresh_signature, {"mode": "created", "notes": []}


def write_scope(cfg, output_dir, fold_ids=None):
    S.atomic_json(
        {
            "updated_at": utc_now(),
            "arms": list(cfg["arms"]),
            "seeds": [int(s) for s in cfg["seeds"]],
            "arm_signatures": {a: arm_signature(cfg, a) for a in cfg["arms"]},
            "arm_recipes": {a: arm_recipe(cfg, a) for a in cfg["arms"]},
            "co_train_group_size": int(cfg["co_train_group_size"]),
            "per_cell_std_floor": configured_std_floor(cfg),
            "tile_budgets": resolve_budgets(cfg),
            "folds_this_session": list(fold_ids or []),
            "note": "arms/seeds/fold selection are not part of the fingerprint; "
                    "they can be extended and resumed at any time.",
        },
        Path(output_dir) / "scope.json",
    )
    S.atomic_json(cfg, Path(output_dir) / "config_used.json")


def mirror_output(output_dir, mirror_dir, skip_weights=False):
    if not mirror_dir:
        return 0
    output_dir = Path(output_dir)
    mirror_dir = Path(mirror_dir)
    copied = 0
    try:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for source in output_dir.rglob("*"):
            relative = source.relative_to(output_dir)
            if relative.parts and relative.parts[0] in MIRROR_SKIP_TOP:
                continue
            if skip_weights and source.name == "inference.pt":
                continue
            destination = mirror_dir / relative
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            try:
                if (destination.exists()
                        and destination.stat().st_size == source.stat().st_size
                        and destination.stat().st_mtime >= source.stat().st_mtime):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied += 1
            except OSError as exc:
                print(f"[MIRROR] Skipped {relative}: {exc}", flush=True)
    except OSError as exc:
        print(f"[MIRROR] Sync failed (training unaffected): {exc}", flush=True)
    if copied:
        print(f"[MIRROR] Synced {copied} file(s) -> {mirror_dir}", flush=True)
    return copied


def refresh_report(cfg, output_dir, mirror_dir, skip_weights=False):
    if cfg.get("write_excel_report", True) and REPORT is not None:
        try:
            path, n = REPORT.build_report(
                output_dir, baseline_arm=cfg.get("report_baseline_arm"),
            )
            print(f"[REPORT] {Path(path).name} updated ({n} completed run(s))",
                  flush=True)
        except Exception as exc:
            print(f"[REPORT] Excel generation failed (training unaffected): "
                  f"{exc}", flush=True)
    mirror_output(output_dir, mirror_dir, skip_weights)


def inspect_bank(bank, cfg, output_dir):
    index = bank.index.copy()

    if index.duplicated(["slide", "row", "col"]).any():
        raise RuntimeError("bank has duplicate physical positions "
                           "(slide/row/col).")
    if index["indication"].map(T.LABEL_MAP).isna().any():
        raise RuntimeError("bank_index.csv has unrecognised labels.")

    if cfg["border_filter_mode"] == "um" and cfg["signed_border_dist_um"] > 0:
        if "dist_border_um" not in index.columns:
            raise RuntimeError("Micron border filtering needs the "
                               "dist_border_um column.")
    if cfg["border_filter_mode"] == "px" and cfg["signed_border_dist_px"] > 0:
        if not {"dist_border_um", "umpp"}.issubset(index.columns):
            raise RuntimeError("Pixel border filtering needs dist_border_um "
                               "and umpp.")

    for slide in index["slide"].astype(str).unique():
        if slide not in bank.shards:
            raise RuntimeError(f"Missing shard for {slide}")
        arr = bank.shards[slide]
        expected_tail = (len(bank.stains), bank.size, bank.size, 3)
        if arr.dtype != np.uint8 or arr.ndim != 5:
            raise RuntimeError(f"{slide} shard has wrong dtype/ndim")
        if tuple(arr.shape[1:]) != expected_tail:
            raise RuntimeError(f"{slide} shard shape disagrees with meta")
        local = index.loc[index["slide"].astype(str) == slide,
                          "local_idx"].to_numpy(np.int64)
        if np.any(local < 0) or np.any(local >= len(arr)):
            raise RuntimeError(f"{slide} local_idx out of range")
        if len(arr):
            sample_rows = sorted(set([0, len(arr) // 2, len(arr) - 1]))
            probe = arr[sample_rows]
            zero_stains = np.all(probe == 0, axis=(0, 2, 3, 4))
            if zero_stains.any():
                names = [bank.stains[i] for i in np.flatnonzero(zero_stains)]
                raise RuntimeError(
                    f"{slide}: {names} are all zero at sampled positions; "
                    "check for missing stains first."
                )

    usable = index.copy()
    if cfg["use_selected_tiles_only"] and "selected" in usable.columns:
        selected = (usable["selected"].astype(str).str.strip().str.lower()
                    .isin(["true", "1", "1.0", "yes", "y", "t"]))
        usable = usable[selected].copy()

    if cfg["border_filter_mode"] == "um":
        usable = T.filter_by_border_um(
            usable, float(cfg["signed_border_dist_um"])).copy()
    elif cfg["border_filter_mode"] == "px":
        usable = T.filter_by_border(
            usable, float(cfg["signed_border_dist_px"])).copy()

    usable = usable.sort_values("bank_idx").reset_index(drop=True)
    if len(usable) == 0:
        raise RuntimeError("No usable tiles after filtering.")

    usable["group"] = usable["slide"].astype(str).map(
        lambda s: group_key_of(s, cfg["split_level"]))
    usable["patient"] = usable["slide"].astype(str).map(T.case_of)
    usable["label"] = usable["indication"].map(T.LABEL_MAP).astype(int)

    rows = []
    for (patient, slide), frame in usable.groupby(["patient", "slide"], sort=True):
        rows.append({
            "patient": str(patient), "slide": str(slide),
            "group_key": str(frame["group"].iloc[0]),
            "usable_tiles": int(len(frame)),
            "cancer_tiles": int((frame["label"] == 1).sum()),
            "non_tumour_tiles": int((frame["label"] == 0).sum()),
            "cancer_fraction": float(frame["label"].mean()),
            "both_classes": bool(frame["label"].nunique() == 2),
            "umpp": (float(pd.to_numeric(frame["umpp"], errors="coerce").median())
                     if "umpp" in frame.columns else float("nan")),
        })
    inventory = pd.DataFrame(rows)
    S.atomic_csv(inventory, Path(output_dir) / "tile_inventory.csv")

    slides_per_patient = inventory.groupby("patient")["slide"].nunique()
    multi = slides_per_patient[slides_per_patient > 1]
    if len(multi):
        warning = (
            "=" * 78
            + "\n[IMPORTANT WARNING] The following case IDs contribute more "
            "than one slide:\n"
            + multi.to_string()
            + "\nThis build splits folds at the slide level only, so different "
              "slides from the same case may appear in both train and test "
              "(adjacent-section leakage).\nIf that matters for your analysis, "
              "make sure each case contributes a single slide to the bank.\n"
            + "=" * 78
        )
        print(warning, flush=True)
        (Path(output_dir) / "WARNINGS.txt").write_text(warning, encoding="utf-8")

    summary = pd.DataFrame([{
        "n_patients": int(usable["patient"].nunique()),
        "n_slides": int(usable["slide"].nunique()),
        "n_groups_for_folds": int(usable["group"].nunique()),
        "split_level": cfg["split_level"],
        "total_bank_tiles": int(len(index)),
        "usable_tiles_after_filtering": int(len(usable)),
        "cancer_tiles": int((usable["label"] == 1).sum()),
        "non_tumour_tiles": int((usable["label"] == 0).sum()),
        "cancer_fraction": float(usable["label"].mean()),
        "tile_um": bank.meta.get("tile_um"),
        "stride_um": bank.meta.get("stride_um"),
        "tile_px": int(bank.size),
        "stains": ";".join(bank.stains),
        "border_filter_mode": cfg["border_filter_mode"],
        "signed_border_dist_um": cfg["signed_border_dist_um"],
        "signed_border_dist_px": cfg["signed_border_dist_px"],
    }])
    S.atomic_csv(summary, Path(output_dir) / "dataset_summary.csv")

    print("\n[DATASET]", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(inventory.to_string(index=False), flush=True)
    return usable


def stratified_cap(frame, maximum, seed, strata_cols):
    frame = frame.sort_values("bank_idx", kind="stable").reset_index(drop=True)
    maximum = int(maximum)
    if len(frame) <= maximum:
        return frame

    key = frame[list(strata_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    levels, inverse = np.unique(key, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    counts = np.bincount(inverse, minlength=len(levels)).astype(np.int64)

    if maximum < len(levels):
        raise ValueError(
            f"tile budget {maximum} is smaller than the number of strata "
            f"{len(levels)} ({list(strata_cols)})."
        )

    target = maximum * counts / counts.sum()
    quota = np.minimum(np.maximum(np.floor(target).astype(np.int64), 1), counts)
    while quota.sum() > maximum:
        eligible = np.flatnonzero(quota > 1)
        if not len(eligible):
            break
        quota[eligible[np.argmax((quota - target)[eligible])]] -= 1
    while quota.sum() < maximum:
        eligible = np.flatnonzero(quota < counts)
        if not len(eligible):
            break
        quota[eligible[np.argmax((target - quota)[eligible])]] += 1

    rng = np.random.default_rng(seed)
    chosen = []
    for level_index, take in enumerate(quota):
        candidates = np.flatnonzero(inverse == level_index)
        chosen.append(rng.choice(candidates, int(take), replace=False))
    order = np.sort(np.concatenate(chosen))
    return frame.iloc[order].reset_index(drop=True)


def apply_budget(frame, budget, seed, strata):
    frame = frame.sort_values("bank_idx", kind="stable").reset_index(drop=True)
    if budget is None:
        return frame
    return stratified_cap(frame, int(budget), seed, strata)


def make_fold_plans(usable, cfg, output_dir):
    level = cfg["split_level"]
    groups = sorted(usable["group"].astype(str).unique())
    if len(groups) < 3:
        raise RuntimeError(f"Need at least 3 {level} groups; only "
                           f"{len(groups)} present.")

    both = {g: usable.loc[usable["group"] == g, "label"].nunique() == 2
            for g in groups}
    budgets = resolve_budgets(cfg)
    strata = {
        "train": ["group", "indication"],
        "validation": ["slide", "indication"],
        "test": ["slide", "indication"],
    }

    folds, composition = [], []
    for outer, test_group in enumerate(groups):
        remaining = [g for g in groups if g != test_group]
        start = outer % len(remaining)
        rotated = remaining[start:] + remaining[:start]
        val_group = next((g for g in rotated if both[g]), None)
        if val_group is None:
            raise RuntimeError(f"{test_group}: no validation {level} with both "
                               "classes present.")

        train_groups = [g for g in remaining if g != val_group]
        selection = {"train": train_groups, "validation": [val_group],
                     "test": [test_group]}

        seed0 = S.stable_seed("fold-budget", level, test_group, val_group)
        packs = {}
        for name, chosen in selection.items():
            frame = usable[usable["group"].isin(chosen)].copy()
            suffix = "val" if name == "validation" else name
            packs[name] = apply_budget(frame, budgets[name],
                                       S.stable_seed(seed0, suffix), strata[name])

        for name in ("train", "validation"):
            if packs[name]["label"].nunique() < 2:
                raise RuntimeError(f"{test_group}: {name} set is missing a "
                                   "class.")
        if len(packs["test"]) == 0:
            raise RuntimeError(f"{test_group}: test set is empty.")

        ids = {k: set(v["bank_idx"].astype(int)) for k, v in packs.items()}
        slides = {k: set(v["slide"].astype(str)) for k, v in packs.items()}
        patients = {k: set(v["patient"].astype(str)) for k, v in packs.items()}
        keys = {k: set(T.key_series(v)) for k, v in packs.items()}
        for a, b in (("train", "validation"), ("train", "test"),
                     ("validation", "test")):
            if ids[a] & ids[b]:
                raise RuntimeError(f"{a} and {b} share a tile.")
            if keys[a] & keys[b]:
                raise RuntimeError(f"{a} and {b} share a physical position.")
            if slides[a] & slides[b]:
                raise RuntimeError(f"{a} and {b} share a slide.")

        test_slides = sorted(packs["test"]["slide"].astype(str).unique())
        val_slides = sorted(packs["validation"]["slide"].astype(str).unique())
        train_slides = sorted(packs["train"]["slide"].astype(str).unique())

        fold_id = f"fold-{outer:03d}"
        plan = {
            "fold_id": fold_id, "split_level": level,
            "test_group": test_group, "val_group": val_group,
            "train_groups": train_groups,
            "test_patients": sorted(patients["test"]),
            "val_patients": sorted(patients["validation"]),
            "train_patients": sorted(patients["train"]),
            "test_slides": test_slides, "val_slides": val_slides,
            "train_slides": train_slides,
            "test_slide": ";".join(test_slides),
            "val_slide": ";".join(val_slides),
            "positions": {k: v["bank_idx"].astype(int).tolist()
                          for k, v in packs.items()},
        }

        plan_path = Path(output_dir) / "splits" / f"{fold_id}.json"
        if plan_path.exists():
            if S.read_json(plan_path) != plan:
                raise RuntimeError(
                    f"{fold_id} saved split differs from the recomputed one. "
                    "Use a new output directory."
                )
        else:
            S.atomic_json(plan, plan_path)

        for name, frame in packs.items():
            saved = frame.copy()
            saved["split"] = name
            saved["fold_id"] = fold_id
            S.atomic_csv(saved,
                         Path(output_dir) / "splits" / f"{fold_id}__{name}.csv")
            composition.append({
                "fold_id": fold_id, "split": name,
                "groups": ";".join(sorted(set(frame["group"].astype(str)))),
                "patients": ";".join(sorted(set(frame["patient"].astype(str)))),
                "slides": ";".join(sorted(set(frame["slide"].astype(str)))),
                "n_slides": int(frame["slide"].nunique()),
                "n_tiles": int(len(frame)),
                "cancer_tiles": int((frame["label"] == 1).sum()),
                "non_tumour_tiles": int((frame["label"] == 0).sum()),
                "cancer_fraction": float(frame["label"].mean()),
            })

        folds.append({**plan, "packs": packs})

    table = pd.DataFrame(composition)
    S.atomic_csv(table, Path(output_dir) / "fold_composition.csv")
    print("\n[FOLDS]", flush=True)
    print(table.to_string(index=False), flush=True)
    return folds


def select_folds(folds, args):
    ids = [f["fold_id"] for f in folds]

    if args.fold_ids:
        wanted = []
        for token in args.fold_ids:
            token = str(token).strip()
            if token.isdigit():
                index = int(token)
                if not 0 <= index < len(folds):
                    raise SystemExit(f"fold index out of range: {token}")
                wanted.append(ids[index])
            else:
                if token not in ids:
                    raise SystemExit(f"unknown fold id: {token} "
                                     f"(available {ids})")
                wanted.append(token)
        chosen = [f for f in folds if f["fold_id"] in set(wanted)]
    elif args.fold_shard:
        text = str(args.fold_shard).replace(" ", "")
        if "/" not in text:
            raise SystemExit("--fold-shard format is i/N, e.g. 1/3")
        i, n = text.split("/", 1)
        i, n = int(i), int(n)
        if not 1 <= i <= n:
            raise SystemExit("--fold-shard requires 1 <= i <= N")
        blocks = np.array_split(np.arange(len(folds)), n)
        chosen = [folds[j] for j in blocks[i - 1]]
    else:
        chosen = list(folds)

    if not chosen:
        raise SystemExit("This session was assigned no folds.")
    print(f"\n[SESSION] This session handles {len(chosen)}/{len(folds)} "
          f"fold(s): {[f['fold_id'] for f in chosen]}", flush=True)
    return chosen


def fixed_groups(cfg):
    by_batch = {}
    for arm in cfg["arms"]:
        by_batch.setdefault(batch_size_for(cfg, arm), []).append(arm)
    groups = []
    limit = int(cfg["co_train_group_size"])
    for _, members in sorted(by_batch.items()):
        groups.extend([members[i:i + limit] for i in range(0, len(members), limit)])
    return groups


def run_id(arm, fold_id, seed):
    return f"{arm}__{fold_id}__seed-{seed}"


def run_directory(output_dir, arm, fold_id, seed):
    return Path(output_dir) / "runs" / run_id(arm, fold_id, seed)


RUN_FILES = ("validation_predictions.csv", "test_predictions.csv",
             "history.csv", "metrics.csv", "inference.pt")


def completed_run(directory, signature, arm_sig):
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    try:
        obj = S.read_json(marker)
    except Exception:
        return False
    if obj.get("signature") != signature:
        raise RuntimeError(f"Completed run has a mismatched experiment "
                           f"fingerprint: {directory}")
    if obj.get("arm_signature") != arm_sig:
        return False
    for filename, expected in obj.get("files", {}).items():
        path = directory / filename
        if not path.exists() or S.file_hash(path) != expected:
            return False
    return True


def audit_parameters(cfg, output_dir):
    rows = []
    for arm_name in cfg["arms"]:
        spec = T.ARM_LIBRARY[arm_name]
        model = make_model(cfg, arm_name, torch.device("cpu"))
        sample = T.make_audit_input(spec, 64, torch.device("cpu"), 1)
        audit = T.parameter_activity(model, sample, include_aux=False)
        rows.append({
            "arm": arm_name, "mode": spec.mode, "readout": spec.readout,
            "n_stains": spec.k,
            "stain_code": bool(arm_model_kwargs(cfg, arm_name)["stain_code"]),
            "readout_groups": spec.n_groups(),
            "input_elements_per_tile": T.input_elements(spec, cfg["tile_size"]),
            "total_params": audit["total_params"],
            "trainable_params": audit["trainable_params"],
            "active_params": audit["active_params"],
            "surviving_batchnorm_layers": T.count_batchnorm(model),
            "arm_signature": arm_signature(cfg, arm_name),
        })
        del model, sample
        gc.collect()

    table = pd.DataFrame(rows)
    reference = float(table["total_params"].median())
    table["params_vs_median_pct"] = (
        (table["total_params"] - reference) / reference * 100.0)
    S.atomic_csv(table, Path(output_dir) / "parameter_audit.csv")
    print("\n[PARAMETER AUDIT]", flush=True)
    print(table.to_string(index=False), flush=True)

    if int(table["surviving_batchnorm_layers"].sum()) != 0:
        raise RuntimeError("An arm still contains BatchNorm; normalisation "
                           "fairness is broken.")


class TrainingTileCache:
    def __init__(self, bank, ids, stop):
        self.ids = np.sort(np.unique(np.asarray(ids, dtype=np.int64)))
        self.stains = list(bank.stains)
        self.stain_idx = dict(bank.stain_idx)
        self.size = bank.size
        self.buf = np.empty(
            (len(self.ids), len(self.stains), self.size, self.size, 3),
            dtype=np.uint8)
        all_stains = list(range(len(self.stains)))
        for start in range(0, len(self.ids), 64):
            stop.check()
            selected = self.ids[start:start + 64]
            self.buf[start:start + len(selected)] = bank.gather(selected, all_stains)

    def gather(self, ids, stain_ids):
        ids = np.asarray(ids, dtype=np.int64)
        loc = np.searchsorted(self.ids, ids)
        if np.any(loc >= len(self.ids)):
            raise IndexError("Position not in the training cache.")
        if not np.array_equal(self.ids[loc], ids):
            raise IndexError("Position not in the training cache.")
        return np.ascontiguousarray(
            self.buf[np.ix_(loc, np.asarray(stain_ids, dtype=np.int64))])


def choose_training_data(bank, frame, cfg, stop):
    if not cfg["cache_training_tiles"]:
        return bank
    need = len(frame) * len(bank.stains) * bank.size * bank.size * 3
    available = psutil.virtual_memory().available
    limit = min(float(cfg["train_cache_max_gb"]) * 1e9,
                available * float(cfg["train_cache_available_fraction"]))
    if need > limit:
        print(f"[TRAIN CACHE] Using mmap: need {need/1e9:.2f}GB, "
              f"allowed {limit/1e9:.2f}GB", flush=True)
        return bank
    try:
        result = TrainingTileCache(bank, frame["bank_idx"].to_numpy(np.int64), stop)
        print(f"[TRAIN CACHE] {need/1e9:.2f}GB uint8 in-memory cache.",
              flush=True)
        return result
    except MemoryError:
        gc.collect()
        print("[TRAIN CACHE] Allocation failed; falling back to mmap.",
              flush=True)
        return bank


class EvalStore:
    """Evaluation cache. Reads only the requested stain columns; can optionally
    reside entirely in GPU memory."""

    def __init__(self, path, stains):
        self.path = Path(path)
        self.stains = list(stains)
        self.arr = np.load(self.path, mmap_mode="r")
        self.gpu = None

    def to_device_cache(self, device, chunk=32):
        if device is None or device.type == "cpu":
            return False
        try:
            buffer = torch.empty(tuple(self.arr.shape), dtype=torch.float32,
                                 device=device)
            for start in range(0, self.arr.shape[0], chunk):
                end = min(start + chunk, self.arr.shape[0])
                buffer[start:end].copy_(
                    torch.from_numpy(np.array(self.arr[start:end],
                                              dtype=np.float32)))
            self.gpu = buffer
            print(f"[EVAL CACHE] {self.path.name} resident in GPU memory "
                  f"({buffer.numel()*4/1e9:.2f}GB)", flush=True)
            return True
        except (RuntimeError, MemoryError) as exc:
            self.gpu = None
            torch.cuda.empty_cache()
            print(f"[EVAL CACHE] GPU residency failed; using disk: {exc}",
                  flush=True)
            return False

    def tensor(self, start, end, requested_stains, device):
        cols = S.stain_columns(self.stains, requested_stains)
        if self.gpu is not None:
            index = torch.as_tensor(cols, dtype=torch.long,
                                    device=self.gpu.device)
            out = self.gpu[start:end].index_select(1, index)
            return out if out.device == device else out.to(device)
        values = np.array(self.arr[start:end, cols], dtype=np.float32, copy=True)
        return torch.from_numpy(values).to(device)

    def chunk(self, start, end):
        return torch.from_numpy(np.array(self.arr[start:end], dtype=np.float32,
                                        copy=True))

    def close(self):
        self.gpu = None
        if self.arr is not None:
            try:
                self.arr._mmap.close()
            except Exception:
                pass
            self.arr = None


def remove_eval_cache(directory, tag):
    directory = Path(directory)
    for suffix in (".npy", ".json", ".building.npy"):
        path = directory / f"{tag}{suffix}"
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


@torch.inference_mode()
def build_eval_store(bank, frame, pipe, cfg, directory, tag, signature, stop,
                     stats_dir=None, rebuild=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{tag}.npy"
    marker = directory / f"{tag}.json"
    temporary = directory / f"{tag}.building.npy"

    if rebuild:
        print(f"[EVAL CACHE] Rebuilding {tag} on request.", flush=True)
        remove_eval_cache(directory, tag)

    ids = frame["bank_idx"].to_numpy(np.int64)
    shape = [len(ids), len(bank.stains), 3,
             int(cfg["tile_size"]), int(cfg["tile_size"])]
    specification = {"signature": signature, "ids": ids.tolist(),
                     "stains": list(bank.stains), "shape": shape,
                     "dtype": "float32"}

    if path.exists() and marker.exists():
        try:
            if S.read_json(marker)["specification"] == specification:
                store = EvalStore(path, bank.stains)
                if (list(store.arr.shape) == shape
                        and store.arr.dtype == np.float32):
                    print(f"[EVAL CACHE] Reusing {path.name}", flush=True)
                    return store
                store.close()
        except (OSError, ValueError, KeyError):
            pass

    required = int(np.prod(shape)) * 4
    if psutil.disk_usage(str(directory)).free < required + 1024 ** 3:
        raise RuntimeError(
            f"Not enough disk space: the {tag} cache needs about "
            f"{required/1e9:.2f}GB (plus headroom)."
        )

    collect_stats = bool(cfg.get("eval_cache_plane_stats", True))
    unit_variance = bool(cfg["per_cell_standardize"])
    floor = effective_std_floor(pipe, cfg)

    arr = None
    plane_rows = []
    try:
        arr = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                        shape=tuple(shape))
        batch = int(cfg["eval_cache_build_batch"])
        all_ids = list(range(len(bank.stains)))

        for start in range(0, len(ids), batch):
            stop.check()
            selected = ids[start:start + batch]
            raw_np = bank.gather(selected, all_ids)
            raw = torch.from_numpy(raw_np)
            values = pipe(raw, train=False, aug=T.AugConfig(), tta_view=0)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("Evaluation preprocessing produced "
                                         "non-finite values.")
            arr[start:start + len(selected)] = values.cpu().numpy()

            if collect_stats:
                out_std = (values.detach().float().cpu()
                           .std(dim=(3, 4), correction=0)
                           .reshape(len(selected), len(bank.stains), 3).numpy())
                raw_f = raw_np.astype(np.float32)
                raw_std = raw_f.std(axis=(2, 3))
                del raw_f
                for i, bank_idx in enumerate(selected):
                    for s_i, stain in enumerate(bank.stains):
                        for c_i in range(3):
                            floored = (unit_variance
                                       and float(out_std[i, s_i, c_i])
                                       < 1.0 - UNIT_STD_TOLERANCE)
                            flat = float(raw_std[i, s_i, c_i]) <= FLAT_RAW_LSB
                            if floored or flat:
                                plane_rows.append({
                                    "bank_idx": int(bank_idx), "stain": stain,
                                    "channel": int(c_i),
                                    "raw_std": float(raw_std[i, s_i, c_i]),
                                    "output_std": float(out_std[i, s_i, c_i]),
                                    "raw_plane_flat": bool(flat),
                                    "standardisation_floor_engaged": bool(floored),
                                })
            del raw, raw_np, values

        arr.flush()
        arr._mmap.close()
        arr = None

        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        S.fsync_directory(directory)
        S.atomic_json({"specification": specification}, marker)
        print(f"[EVAL CACHE] Built {tag}: {required/1e9:.2f}GB.", flush=True)

        if collect_stats and stats_dir is not None:
            stats = pd.DataFrame(plane_rows)
            S.atomic_csv(stats, Path(stats_dir) / f"{tag}_plane_stats.csv")
            total_planes = len(ids) * len(bank.stains) * 3
            if len(stats):
                per_stain = (stats.groupby("stain")["standardisation_floor_engaged"]
                             .agg(["size", "sum"])
                             .rename(columns={"size": "flagged_planes",
                                              "sum": "floored_planes"}))
                print(f"[EVAL CACHE] {tag}: {len(stats)}/{total_planes} plane(s) "
                      f"are blank or hit the standardisation floor "
                      f"(floor={floor:g}):", flush=True)
                print(per_stain.to_string(), flush=True)
            else:
                print(f"[EVAL CACHE] {tag}: no blank/floored planes.",
                      flush=True)

        return EvalStore(path, bank.stains)
    finally:
        if arr is not None:
            try:
                arr._mmap.close()
            except Exception:
                pass
        if temporary.exists():
            temporary.unlink()


@torch.inference_mode()
def check_eval_store(bank, frame, pipe, store, cfg, device, stop, tag,
                     report_dir, mode="strict", probe_chunks=1,
                     rel_tol=None, corr_tol=None):
    """Triple, condition-number-aware cache check (builder replay / TTA
    equivariance / stain-column selection)."""
    if mode not in CACHE_CHECK_MODES:
        raise ValueError(f"Unknown cache-check mode: {mode}")
    if mode == "off":
        print(f"[EVAL CACHE CHECK] {tag}: skipped (--cache-check off).",
              flush=True)
        return None
    if store is None or len(frame) == 0:
        return None

    report_dir = Path(report_dir)
    ids = frame["bank_idx"].to_numpy(np.int64)
    build_batch = int(cfg["eval_cache_build_batch"])
    n_chunks = int(math.ceil(len(ids) / build_batch))
    wanted = max(1, min(int(probe_chunks), n_chunks))
    picks = sorted(set(np.linspace(0, n_chunks - 1, wanted)
                       .round().astype(int).tolist()))

    all_cols = list(range(len(bank.stains)))
    views = T.TTA_VIEWS[:max(int(cfg["tta_epoch"]), int(cfg["tta_final"]))]
    unit_variance = bool(cfg["per_cell_standardize"])
    floor = effective_std_floor(pipe, cfg)

    requests = {}
    for arm_name in cfg["arms"]:
        requests.setdefault(tuple(T.ARM_LIBRARY[arm_name].stains), []).append(arm_name)

    def annotate(report, kind, chunk, view, arm, names, positions):
        report = report.copy()
        report["check"] = kind
        report["chunk"] = int(chunk)
        report["view"] = int(view)
        report["arm"] = arm
        report["stains"] = ";".join(names)
        report["bank_idx"] = np.asarray(positions)[report["position"].to_numpy(int)]
        report["stain"] = [names[i] for i in report["stain_slot"].to_numpy(int)]
        return report

    reports = []
    for chunk in picks:
        stop.check()
        start = chunk * build_batch
        end = min(start + build_batch, len(ids))
        positions = ids[start:end]

        raw_np = bank.gather(positions, all_cols)
        raw = torch.from_numpy(raw_np)

        direct = pipe(raw, train=False, aug=T.AugConfig(), tta_view=0).float().cpu()
        cached = store.chunk(start, end)
        reports.append(annotate(
            plane_agreement(direct, cached, raw_np, unit_variance),
            "builder_replay", chunk, 0, "(all stains)", list(bank.stains), positions))
        del direct

        for view in views[1:]:
            stop.check()
            rotated_direct = pipe(raw, train=False, aug=T.AugConfig(),
                                  tta_view=view).float().cpu()
            rotated_cached = T._dihedral(cached, view)
            reports.append(annotate(
                plane_agreement(rotated_direct, rotated_cached, raw_np,
                                unit_variance),
                "tta_equivariance", chunk, view, "(all stains)",
                list(bank.stains), positions))
            del rotated_direct, rotated_cached
        del cached

        for requested, arm_names in requests.items():
            stop.check()
            cols = S.stain_columns(bank.stains, list(requested))
            sub_np = np.ascontiguousarray(raw_np[:, cols])
            sub_direct = pipe(torch.from_numpy(sub_np), train=False,
                              aug=T.AugConfig(), tta_view=0).float().cpu()
            sub_cached = store.tensor(start, end, list(requested),
                                      torch.device("cpu")).float().cpu()
            reports.append(annotate(
                plane_agreement(sub_direct, sub_cached, sub_np, unit_variance),
                "stain_selection", chunk, 0, arm_names[0],
                list(requested), positions))
            del sub_direct, sub_cached, sub_np

        del raw, raw_np
        gc.collect()

    report = pd.concat(reports, ignore_index=True)

    summaries, failures, suspicious = {}, [], []
    hard = False
    for kind, policy in CHECK_POLICY.items():
        part = report.loc[report["check"] == kind]
        if not len(part):
            continue
        verdict = cache_check_verdict(
            part,
            rel_tol=policy["rel_tol"] if rel_tol is None else float(rel_tol),
            corr_tol=policy["corr_tol"] if corr_tol is None else float(corr_tol),
            degenerate_rel_tol=policy["degenerate_rel_tol"])
        summaries[kind] = {k: v for k, v in verdict.items()
                           if k not in ("failures", "suspicious")}
        if len(verdict["failures"]):
            failures.append(verdict["failures"])
        if len(verdict["suspicious"]):
            suspicious.append(verdict["suspicious"])
        hard = hard or verdict["hard_failures"]
        print(f"[EVAL CACHE CHECK] {tag} | {kind:<16s} "
              f"{verdict['n_planes']:>6d} planes, {verdict['n_checked']:>6d} checked, "
              f"max rel {verdict['max_rel_diff']:.1e}, "
              f"min corr {verdict['min_corr']:.6f}, "
              f"{verdict['n_degenerate']} degenerate, "
              f"{verdict['n_failures']} failure(s)", flush=True)

    failure_table = (pd.concat(failures, ignore_index=True) if failures
                     else report.iloc[0:0])
    suspicious_table = (pd.concat(suspicious, ignore_index=True) if suspicious
                        else report.iloc[0:0])

    S.atomic_csv(report, report_dir / f"eval_cache_check__{tag}.csv")
    S.atomic_json({
        "tag": tag, "mode": mode, "checked_at": utc_now(),
        "probe_chunks": picks, "build_batch": build_batch,
        "per_cell_std_floor": floor,
        "ulp_amplification_on_blank_planes": 1.0 / floor,
        "rel_tol_override": rel_tol, "corr_tol_override": corr_tol,
        "policy": CHECK_POLICY, "summaries": summaries,
        "hard_failures": bool(hard),
    }, report_dir / f"eval_cache_check__{tag}.json")

    total_degenerate = sum(v["n_degenerate"] for v in summaries.values())
    if total_degenerate:
        print(f"[EVAL CACHE CHECK] {tag}: {total_degenerate} degenerate plane(s) "
              f"excluded from the numeric gate (blank tiles whose true std is "
              f"below per_cell_std_floor={floor:g}).", flush=True)

    if len(suspicious_table):
        print(f"[EVAL CACHE CHECK][WARNING] {len(suspicious_table)} plane(s) hit "
              "the floor but the raw tile is not flat; please inspect "
              "manually:", flush=True)
        print(suspicious_table[["check", "arm", "bank_idx", "stain", "channel",
                                "raw_std", "a_std", "b_std"]]
              .head(10).to_string(index=False), flush=True)

    if hard:
        detail = failure_table[["check", "arm", "bank_idx", "stain", "channel",
                                "rel_diff", "corr", "a_std", "b_std", "raw_std",
                                "degenerate"]].head(10).to_string(index=False)
        message = (
            f"{tag} evaluation cache check failed: {len(failure_table)} plane(s) "
            f"out of tolerance.\n{detail}\n"
            f"Full report: {report_dir / f'eval_cache_check__{tag}.csv'}\n"
            "Low corr = genuinely the wrong stain; high corr but large rel_diff "
            "= numeric issue; builder_replay failure = the cache file itself is "
            "corrupt, pass --rebuild-eval-cache."
        )
        if mode == "strict":
            raise RuntimeError(message)
        print("[EVAL CACHE CHECK][WARNING] " + message, flush=True)
        S.atomic_json({"tag": tag, "recorded_at": utc_now(),
                       "n_failures": int(len(failure_table))},
                      report_dir / f"FAILED__{tag}.json")
    else:
        print(f"[EVAL CACHE CHECK] {tag}: passed (cache replay / TTA "
              "equivariance / stain-column selection).", flush=True)
    return summaries


@torch.inference_mode()
def predict(model, arm_name, frame, bank, pipe, store,
            cfg, device, amp_on, amp_dtype, n_views, stop):
    views = T.TTA_VIEWS[:int(n_views)]
    if len(views) != int(n_views) or not views:
        raise ValueError("Invalid number of TTA views.")

    model.eval()
    requested = T.ARM_LIBRARY[arm_name].stains
    global_cols = S.stain_columns(bank.stains, requested)
    ids = frame["bank_idx"].to_numpy(np.int64)
    logits = np.empty(len(frame), dtype=np.float64)
    batch = int(cfg["eval_batch_size"])

    for start in range(0, len(frame), batch):
        stop.check()
        end = min(start + batch, len(frame))

        if store is not None:
            cached = store.tensor(start, end, requested, device)
            raw = None
        else:
            raw = torch.from_numpy(bank.gather(ids[start:end], global_cols))
            cached = None

        accumulated = None
        for view in views:
            if cached is not None:
                inputs = T._dihedral(cached, view) if view else cached
            else:
                inputs = pipe(raw, train=False, aug=T.AugConfig(), tta_view=view)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_on):
                current = model(inputs).float().reshape(-1)
            accumulated = current if accumulated is None else accumulated + current

        values = (accumulated / len(views)).cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{arm_name} produced a non-finite logit")
        logits[start:end] = values

    return logits


class ResumableBatches:
    def __init__(self, data, frame, weights, stains, batch, order, start_batch):
        self.data = data
        self.ids = frame["bank_idx"].to_numpy(np.int64)
        self.labels = frame["label"].to_numpy(np.int64)
        self.weights = np.asarray(weights, np.float32)
        self.meta = frame
        self.stain_ids = S.stain_columns(data.stains, stains)
        self.batch = int(batch)
        self.order = np.asarray(order, np.int64)
        self.start_batch = int(start_batch)
        self.n = len(frame)

    def __len__(self):
        return max(0, math.ceil(self.n / self.batch) - self.start_batch)

    def __iter__(self):
        total = math.ceil(self.n / self.batch)
        for batch_index in range(self.start_batch, total):
            selected = self.order[batch_index * self.batch:
                                  (batch_index + 1) * self.batch]
            raw = self.data.gather(self.ids[selected], self.stain_ids)
            yield (torch.from_numpy(raw),
                   torch.from_numpy(self.labels[selected].astype(np.float32)),
                   torch.from_numpy(self.weights[selected]),
                   batch_index)


def training_weights(frame, level_col):
    counts = frame[level_col].astype(str).value_counts()
    weights = frame[level_col].astype(str).map(
        lambda x: 1.0 / counts[x]).to_numpy(np.float32)
    return weights / weights.mean()


def make_order(n, seed, epoch):
    order = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed) * 100003 + int(epoch))
    rng.shuffle(order)
    return order


class RunState:
    CONTROL_KEYS = ("history", "pool", "best", "no_improve", "done",
                    "stopped_early", "last_epoch", "n_examples")

    def __init__(self, cfg, arm_name, seed, labels, weights, device):
        self.name = arm_name
        self.device = device
        self.cfg = cfg
        self.batch = batch_size_for(cfg, arm_name)

        T.set_seed(seed, cfg["strict_determinism"])
        self.model = make_model(cfg, arm_name, device)

        scale = (self.batch / float(cfg["lr_ref_batch"])
                 if cfg["scale_lr_with_batch"] else 1.0)
        body, head = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                (head if name.startswith(("head.", "aux.")) else body).append(parameter)

        groups = [{"params": body, "lr": cfg["lr"] * scale},
                  {"params": head, "lr": cfg["lr"] * scale * cfg["head_lr_mult"]}]
        self.optimizer = torch.optim.AdamW(groups,
                                           weight_decay=cfg["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: T.lr_lambda(epoch, cfg["epochs"], cfg["warmup_epochs"]))

        self.ema = (S.EMA(self.model, cfg["ema_decay"], cfg["fast_ema"])
                    if cfg["ema_decay"] > 0 else None)

        self.amp_on, self.amp_dtype = T.amp_dtype_for(device, cfg["use_amp"])
        self.scaler = None
        if self.amp_on and device.type == "cuda" and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda")

        positive = float(weights[labels == 1].sum())
        negative = float(weights[labels == 0].sum())
        if positive <= 0 or negative <= 0:
            raise RuntimeError("Training needs both classes present.")
        self.pos_weight = torch.tensor([negative / positive],
                                       dtype=torch.float32, device=device)

        self.history = []
        self.pool = []
        self.best = -float("inf")
        self.no_improve = 0
        self.done = False
        self.stopped_early = False
        self.last_epoch = 0
        self.reset_epoch()

    def reset_epoch(self):
        self.loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.n_examples = 0

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": None if self.ema is None else self.ema.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "loss_sum": self.loss_sum,
            "control": {k: getattr(self, k) for k in self.CONTROL_KEYS},
        }

    def load_state_dict(self, saved):
        self.model.load_state_dict(saved["model"])
        self.scheduler.load_state_dict(saved["scheduler"])
        self.optimizer.load_state_dict(saved["optimizer"])
        if self.ema is not None:
            if saved["ema"] is None:
                raise RuntimeError("Checkpoint is missing EMA state.")
            self.ema.load_state_dict(saved["ema"], self.device)
        if self.scaler is not None:
            self.scaler.load_state_dict(saved["scaler"])
        self.loss_sum = saved["loss_sum"].to(self.device)
        for key, value in saved["control"].items():
            setattr(self, key, value)

    def evaluation_model(self, final=False):
        model = copy.deepcopy(self.model)
        if final and self.pool:
            averaged = T.average_states([item[2] for item in self.pool])
            model.load_state_dict(averaged)
        elif self.ema is not None:
            self.ema.copy_to(model)
        model.eval()
        return model


def save_run_outputs(st, fold, seed, bank, pipe, stores, cfg, device,
                     output_dir, signature, progress, stop):
    directory = run_directory(output_dir, st.name, fold["fold_id"], seed)
    directory.mkdir(parents=True, exist_ok=True)

    model = st.evaluation_model(final=True)
    try:
        predictions, metrics_by_split = {}, {}
        val_frame = fold["packs"]["validation"]
        val_labels = val_frame["label"].to_numpy(np.int64)

        val_logits = predict(model, st.name, val_frame, bank, pipe,
                             stores.get("validation"), cfg, device,
                             st.amp_on, st.amp_dtype, cfg["tta_final"], stop)
        val_prob = T.sigmoid(val_logits)
        threshold, repaired = T.youden_threshold(val_labels, val_prob)

        for split in ("validation", "test"):
            frame = fold["packs"][split]
            labels = frame["label"].to_numpy(np.int64)
            if split == "validation":
                logits, probabilities = val_logits, val_prob
            else:
                logits = predict(model, st.name, frame, bank, pipe,
                                 stores.get(split), cfg, device,
                                 st.amp_on, st.amp_dtype, cfg["tta_final"], stop)
                probabilities = T.sigmoid(logits)

            metrics_by_split[split] = checked_metrics(labels, probabilities,
                                                      threshold)
            metadata_cols = [c for c in ("bank_idx", "slide", "patient", "group",
                                         "indication", "row", "col", "x", "y",
                                         "umpp", "dist_border_um")
                             if c in frame.columns]
            pred = frame[metadata_cols].copy()
            pred["run_id"] = run_id(st.name, fold["fold_id"], seed)
            pred["fold_id"] = fold["fold_id"]
            pred["split_level"] = cfg["split_level"]
            pred["arm"] = st.name
            pred["seed"] = int(seed)
            pred["test_group"] = fold["test_group"]
            pred["val_group"] = fold["val_group"]
            pred["test_slide"] = fold["test_slide"]
            pred["val_slide"] = fold["val_slide"]
            pred["split"] = split
            pred["label"] = labels
            pred["logit"] = logits
            pred["prob"] = probabilities
            pred["threshold"] = float(threshold)
            predictions[split] = pred

        benchmark = {}
        if cfg["benchmark_repeats"] > 0:
            stop.check()
            n = min(8, len(val_frame))
            requested = T.ARM_LIBRARY[st.name].stains
            store = stores.get("validation")
            if store is not None:
                sample = store.tensor(0, n, requested, device)
            else:
                gids = S.stain_columns(bank.stains, requested)
                raw = torch.from_numpy(bank.gather(
                    val_frame["bank_idx"].to_numpy(np.int64)[:n], gids))
                sample = pipe(raw, train=False, aug=T.AugConfig(), tta_view=0)
            bench_model = make_benchmark_model(cfg, st.name, device,
                                               model.state_dict())
            try:
                benchmark = T.benchmark_model(
                    bench_model, sample, device, warmup=3,
                    repeats=cfg["benchmark_repeats"],
                    amp_on=st.amp_on, amp_dtype=st.amp_dtype)
            finally:
                del bench_model
            del sample

        selected = ([int(item[1]) for item in st.pool] if st.pool
                    else [int(st.last_epoch)])
        total_params, trainable_params = T.count_params(model)
        spec = T.ARM_LIBRARY[st.name]

        record = {
            "run_id": run_id(st.name, fold["fold_id"], seed),
            "fold_id": fold["fold_id"], "arm": st.name, "seed": int(seed),
            "split_level": cfg["split_level"],
            "test_group": fold["test_group"], "val_group": fold["val_group"],
            "test_slide": fold["test_slide"], "val_slide": fold["val_slide"],
            "n_test_slides": len(fold["test_slides"]),
            "n_train": len(fold["packs"]["train"]), "n_val": len(val_frame),
            "n_test": len(fold["packs"]["test"]),
            "train_pos_frac": float(fold["packs"]["train"]["label"].mean()),
            "test_pos_frac": float(fold["packs"]["test"]["label"].mean()),
            "threshold": float(threshold),
            "threshold_repaired": bool(repaired),
            "threshold_method": "validation_youden",
            "val_auroc": metrics_by_split["validation"]["auroc"],
            "epochs_run": int(st.last_epoch),
            "stopped_early": bool(st.stopped_early),
            "selected_epochs": ";".join(map(str, selected)),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "input_elements_per_tile": T.input_elements(spec, cfg["tile_size"]),
            "batch_size": int(st.batch), "n_input_stains": spec.k,
            "arm_mode": spec.mode, "arm_readout": spec.readout,
            "arm_signature": arm_signature(cfg, st.name),
            "group_active_wall_s": float(progress["active_wall_s"]),
            "timing_scope": "shared_group_not_individual_training_cost",
            "model_latency_ms_per_location": benchmark.get(
                "model_latency_ms_per_location", float("nan")),
            "model_locations_per_second": benchmark.get(
                "model_locations_per_second", float("nan")),
            "model_peak_memory_bytes": benchmark.get(
                "model_peak_memory_bytes", float("nan")),
            "benchmark_scope": "canonical_forward_fast_gap_disabled",
        }
        record.update({"test_" + k: v
                       for k, v in metrics_by_split["test"].items()})
        record.update({"valm_" + k: v
                       for k, v in metrics_by_split["validation"].items()})

        for split, frame in predictions.items():
            S.atomic_csv(frame, directory / f"{split}_predictions.csv")
        S.atomic_csv(pd.DataFrame(st.history), directory / "history.csv")
        S.atomic_csv(pd.DataFrame([record]), directory / "metrics.csv")

        S.atomic_checkpoint({
            "signature": signature,
            "arm_signature": arm_signature(cfg, st.name),
            "run_id": record["run_id"], "arm": st.name,
            "arm_spec": {"mode": spec.mode, "stains": list(spec.stains),
                         "readout": spec.readout},
            "backbone": cfg["backbone"],
            "model_kwargs": arm_model_kwargs(cfg, st.name),
            "preprocessing": {
                "tile_size": cfg["tile_size"], "color_mode": cfg["color_mode"],
                "per_cell_standardize": cfg["per_cell_standardize"],
                "per_cell_std_floor": effective_std_floor(pipe, cfg),
                "pretrained": cfg["pretrained"], "tta_final": cfg["tta_final"],
                "groupnorm_2d": bool(T.USE_GROUPNORM_2D),
            },
            "threshold": float(threshold), "selected_epochs": selected,
            "state": model.state_dict(),
        }, directory / "inference.pt")

        S.atomic_json({
            "run_id": record["run_id"], "recorded_at": utc_now(),
            "gpu_name": gpu_name(), "device_type": device.type,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "amp_dtype": str(st.amp_dtype), "amp_enabled": bool(st.amp_on),
            "per_cell_std_floor": effective_std_floor(pipe, cfg),
            "tile_budgets": resolve_budgets(cfg),
            "co_train_group_size": int(cfg["co_train_group_size"]),
        }, directory / "run_environment.json")

        S.atomic_json({
            "signature": signature,
            "arm_signature": arm_signature(cfg, st.name),
            "run_id": record["run_id"], "completed_at": utc_now(),
            "files": {name: S.file_hash(directory / name) for name in RUN_FILES},
        }, directory / "complete.json")

        print(f"[COMPLETE] {record['run_id']} "
              f"AUROC={fmt(record['test_auroc'])} "
              f"Acc={fmt(record['test_accuracy'])} "
              f"F1={fmt(record['test_f1'])} "
              f"Brier={fmt(record['test_brier'])}", flush=True)
    finally:
        del model


def write_run_index(cfg, folds, output_dir, signature, all_folds=None):
    rows = []
    for fold in (all_folds or folds):
        for arm in cfg["arms"]:
            arm_sig = arm_signature(cfg, arm)
            for seed in cfg["seeds"]:
                directory = run_directory(output_dir, arm, fold["fold_id"], seed)
                rows.append({
                    "arm": arm, "fold_id": fold["fold_id"], "seed": int(seed),
                    "test_group": fold["test_group"],
                    "test_slide": fold["test_slide"],
                    "complete": bool(completed_run(directory, signature, arm_sig)),
                    "path": str(directory),
                })
    table = pd.DataFrame(rows)
    S.atomic_csv(table, Path(output_dir) / "run_index.csv")
    done = int(table["complete"].sum())
    S.atomic_json({
        "updated_at": utc_now(), "completed_runs": done,
        "expected_runs": len(table),
        "fraction_complete": float(done / max(len(table), 1)),
        "split_level": cfg["split_level"],
        "note": "Use the analyze_results script for statistics and plots.",
    }, Path(output_dir) / "training_status.json")
    print(f"[PROGRESS] {done}/{len(table)} run(s) completed.", flush=True)
    return table


def train_group(group, group_index, fold, seed, bank, training_data,
                pipe, stores, cfg, device, output_dir, signature, stop,
                save_every_batches, save_every_seconds):
    arm_sigs = {arm: arm_signature(cfg, arm) for arm in group}

    if all(completed_run(run_directory(output_dir, arm, fold["fold_id"], seed),
                         signature, arm_sigs[arm]) for arm in group):
        print(f"[SKIP COMPLETE] {fold['fold_id']} seed={seed} group={group}",
              flush=True)
        return

    train_frame = fold["packs"]["train"]
    y_train = train_frame["label"].to_numpy(np.int64)
    weights = training_weights(train_frame, "group")
    batch = batch_size_for(cfg, group[0])
    if any(batch_size_for(cfg, arm) != batch for arm in group):
        raise RuntimeError("All arms in a co-training group must use the same "
                           "batch size.")

    if cfg["shared_full_stain_batch"]:
        batch_stains = list(bank.stains)
    else:
        requested = {s for arm in group for s in T.ARM_LIBRARY[arm].stains}
        batch_stains = [s for s in bank.stains if s in requested]

    states = [RunState(cfg, arm, seed, y_train, weights, device) for arm in group]
    train_cols = {
        st.name: torch.tensor(
            S.stain_columns(batch_stains, T.ARM_LIBRARY[st.name].stains),
            dtype=torch.long, device=device)
        for st in states
    }

    checkpoint_path = (Path(output_dir) / "progress"
                       / f"{fold['fold_id']}__seed-{seed}__group-{group_index:02d}.pt")
    loaded = S.load_progress(checkpoint_path, signature)
    is_new_group = loaded is None

    if is_new_group:
        if any((run_directory(output_dir, arm, fold["fold_id"], seed)
                / "complete.json").exists() for arm in group):
            if all(completed_run(
                    run_directory(output_dir, arm, fold["fold_id"], seed),
                    signature, arm_sigs[arm]) for arm in group):
                print("[SKIP COMPLETE] Verified completed group.", flush=True)
                return
        T.set_seed(S.stable_seed("group-training", seed, fold["fold_id"]),
                   cfg["strict_determinism"])
        progress = {
            "group": list(group), "fold_id": fold["fold_id"], "seed": int(seed),
            "epoch": 1, "next_batch": 0, "validation_cursor": 0,
            "phase": "train",
            "order": torch.from_numpy(make_order(len(train_frame), seed, 1)),
            "global_batches": 0, "active_wall_s": 0.0,
        }
    else:
        progress = loaded["progress"]
        if progress["group"] != list(group):
            raise RuntimeError(
                "The saved checkpoint's group membership changed. Restore the "
                "original co_train_group_size, or delete this progress file to "
                "retrain the group.")
        for st in states:
            st.load_state_dict(loaded["arms"][st.name])
        try:
            S.restore_rng(loaded["rng"], device)
        except Exception as exc:
            print(f"[RESUME] Skipping RNG restore (harmless: every batch is "
                  f"deterministically re-seeded): {exc}", flush=True)
        print(f"[RESUME] {checkpoint_path.name}: phase={progress['phase']} "
              f"epoch={progress['epoch']} next_batch={progress['next_batch']}",
              flush=True)
        del loaded

    last_clock = time.monotonic()
    last_save_clock = last_clock
    last_saved_batch = int(progress["global_batches"])

    def account_time():
        nonlocal last_clock
        now = time.monotonic()
        progress["active_wall_s"] += now - last_clock
        last_clock = now

    def persist(force=False):
        nonlocal last_save_clock, last_saved_batch
        due_batches = (progress["global_batches"] - last_saved_batch
                       >= save_every_batches)
        due_time = time.monotonic() - last_save_clock >= save_every_seconds
        if not (force or due_batches or due_time):
            return

        T.synchronize_device(device)
        account_time()
        payload = {
            "signature": signature, "arm_signatures": arm_sigs,
            "saved_at": utc_now(), "progress": progress,
            "arms": {st.name: st.state_dict() for st in states},
            "rng": S.capture_rng(device),
        }
        S.atomic_checkpoint(payload, checkpoint_path)
        S.atomic_json({
            "checkpoint": str(checkpoint_path), "saved_at": utc_now(),
            "fold_id": fold["fold_id"], "seed": int(seed), "group": group,
            "phase": progress["phase"], "epoch": progress["epoch"],
            "next_batch": progress["next_batch"],
            "validation_cursor": progress["validation_cursor"],
            "global_batches": progress["global_batches"],
        }, Path(output_dir) / "current_progress.json")

        last_save_clock = time.monotonic()
        last_saved_batch = int(progress["global_batches"])
        print(f"[SAVED] {checkpoint_path.name} phase={progress['phase']} "
              f"epoch={progress['epoch']} next_batch={progress['next_batch']}",
              flush=True)

    if is_new_group:
        persist(force=True)

    aug = T.AugConfig(**cfg["aug"])

    try:
        while progress["phase"] != "final":
            stop.check()
            epoch = int(progress["epoch"])

            if progress["phase"] == "train":
                active = [st for st in states if not st.done]
                if not active:
                    progress["phase"] = "final"
                    persist(force=True)
                    break
                for st in active:
                    st.model.train()

                iterator = ResumableBatches(
                    training_data, train_frame, weights, batch_stains, batch,
                    progress["order"].cpu().numpy(), progress["next_batch"])
                stream = (T.Prefetcher(iterator, depth=cfg["prefetch_depth"])
                          if cfg["prefetch_depth"] > 0 else iterator)
                stream_iterator = iter(stream)
                epoch_start = time.monotonic()

                try:
                    for raw, labels, sample_weights, batch_index in stream_iterator:
                        stop.check()

                        S.seed_torch(S.stable_seed(
                            "aug", fold["fold_id"], seed, epoch, batch_index))
                        full = pipe(raw, train=cfg["augment"], aug=aug, rng=None)

                        targets = labels.to(device).unsqueeze(1)
                        sample_weights = sample_weights.to(device).unsqueeze(1)

                        for st in active:
                            S.seed_torch(S.stable_seed(
                                "arm", st.name, fold["fold_id"], seed,
                                epoch, batch_index))
                            inputs = full.index_select(1, train_cols[st.name])
                            if cfg["augment"]:
                                inputs = T.modality_dropout(inputs, aug)

                            st.optimizer.zero_grad(set_to_none=True)
                            with torch.autocast(device_type=device.type,
                                                dtype=st.amp_dtype,
                                                enabled=st.amp_on):
                                logits = st.model(inputs)
                                loss = T.weighted_bce(
                                    logits, targets, sample_weights,
                                    cfg["label_smoothing"], st.pos_weight)

                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError(
                                    f"{st.name}: training loss is non-finite. "
                                    "The previous checkpoint is kept.")

                            if st.scaler is not None:
                                st.scaler.scale(loss).backward()
                                st.scaler.unscale_(st.optimizer)
                            else:
                                loss.backward()

                            total_norm = torch.nn.utils.clip_grad_norm_(
                                st.model.parameters(), cfg["grad_clip"],
                                error_if_nonfinite=False, foreach=True)

                            if torch.isfinite(total_norm):
                                if st.scaler is not None:
                                    st.scaler.step(st.optimizer)
                                    st.scaler.update()
                                else:
                                    st.optimizer.step()
                                if st.ema is not None:
                                    st.ema.update(st.model)
                            else:
                                st.optimizer.zero_grad(set_to_none=True)
                                if st.scaler is not None:
                                    st.scaler.update()
                                print(f"[NONFINITE-GRAD] skipped: {st.name} "
                                      f"fold={fold['fold_id']} epoch={epoch} batch={batch_index}",
                                      flush=True)

                            st.loss_sum += loss.detach().float() * len(labels)
                            st.n_examples += len(labels)
                            st.optimizer.zero_grad(set_to_none=True)
                            del inputs, logits, loss

                        del full
                        progress["next_batch"] = int(batch_index) + 1
                        progress["global_batches"] += 1
                        stop.after_batch()
                        persist()
                        stop.check()
                finally:
                    if hasattr(stream_iterator, "close"):
                        stream_iterator.close()

                print(f"[TIMING] {fold['fold_id']} seed={seed} epoch={epoch} "
                      f"train time {time.monotonic() - epoch_start:.1f}s", flush=True)
                progress["phase"] = "validate"
                progress["validation_cursor"] = 0
                persist()

            if progress["phase"] == "validate":
                val_frame = fold["packs"]["validation"]
                val_labels = val_frame["label"].to_numpy(np.int64)

                while progress["validation_cursor"] < len(states):
                    stop.check()
                    j = int(progress["validation_cursor"])
                    st = states[j]

                    if not st.done:
                        evaluation_model = st.evaluation_model(final=False)
                        try:
                            logits = predict(
                                evaluation_model, st.name, val_frame, bank, pipe,
                                stores.get("validation"), cfg, device,
                                st.amp_on, st.amp_dtype, cfg["tta_epoch"], stop)
                            vauc = float(T.roc_auc_score(val_labels, logits))
                            loss_mean = (float(st.loss_sum.cpu().item()) / st.n_examples
                                         if st.n_examples else float("nan"))
                            if not np.isfinite(loss_mean):
                                raise FloatingPointError("No valid training loss "
                                                         "for this epoch.")

                            lr_used = float(st.optimizer.param_groups[0]["lr"])
                            st.scheduler.step()
                            st.last_epoch = epoch
                            st.history.append({
                                "arm": st.name, "fold_id": fold["fold_id"],
                                "seed": int(seed), "epoch": epoch,
                                "train_loss": loss_mean, "val_auroc": vauc,
                                "lr_body_used": lr_used,
                                "lr_body_next": float(
                                    st.optimizer.param_groups[0]["lr"]),
                            })

                            recent = [r["val_auroc"]
                                      for r in st.history[-cfg["val_smooth"]:]]
                            smooth = float(np.mean(recent))

                            if epoch >= cfg["min_epochs"]:
                                st.pool.append((smooth, epoch,
                                                S.snapshot_state(evaluation_model)))
                                st.pool.sort(key=lambda item: (-item[0], item[1]))
                                st.pool = st.pool[:cfg["swa_top_k"]]
                                if smooth > st.best + cfg["min_delta"]:
                                    st.best = smooth
                                    st.no_improve = 0
                                else:
                                    st.no_improve += 1
                                if st.no_improve >= cfg["patience"]:
                                    st.done = True
                                    st.stopped_early = True

                            print(f"[EPOCH] {fold['fold_id']} seed={seed} "
                                  f"{st.name} {epoch}/{cfg['epochs']} "
                                  f"loss={fmt(loss_mean)} val_AUROC={fmt(vauc)} "
                                  f"early_stop={st.stopped_early}", flush=True)
                        finally:
                            del evaluation_model

                    progress["validation_cursor"] = j + 1
                    persist()

                if epoch >= cfg["epochs"] or all(st.done for st in states):
                    progress["phase"] = "final"
                else:
                    progress["epoch"] = epoch + 1
                    progress["next_batch"] = 0
                    progress["validation_cursor"] = 0
                    progress["phase"] = "train"
                    progress["order"] = torch.from_numpy(
                        make_order(len(train_frame), seed, epoch + 1))
                    for st in states:
                        if not st.done:
                            st.reset_epoch()
                persist()

        for st in states:
            stop.check()
            directory = run_directory(output_dir, st.name, fold["fold_id"], seed)
            if completed_run(directory, signature, arm_sigs[st.name]):
                continue
            account_time()
            save_run_outputs(st, fold, seed, bank, pipe, stores, cfg, device,
                             output_dir, signature, progress, stop)

        if cfg["delete_group_checkpoint_when_done"] and all(
                completed_run(run_directory(output_dir, arm, fold["fold_id"], seed),
                              signature, arm_sigs[arm]) for arm in group):
            S.remove_checkpoint(checkpoint_path)
            print(f"[CLEANUP] Removed {checkpoint_path.name} (group fully "
                  "complete).", flush=True)

    except PauseRequested:
        persist(force=True)
        print("[PAUSED] Training state saved safely.", flush=True)
        raise
    finally:
        states.clear()
        release_device(device)


def want_val_cache_on_gpu(cfg, device, store):
    mode = cfg.get("val_cache_on_gpu", "auto")
    if mode == "off" or device.type != "cuda" or store is None:
        return False
    size = int(np.prod(store.arr.shape)) * 4
    free, total = torch.cuda.mem_get_info()
    if mode == "on":
        return size < free * 0.8
    return (size <= total * float(cfg.get("val_cache_gpu_max_fraction", 0.25))
            and size < free * 0.5)


def execute(cfg, bank, device, signature, output_dir, args, stop):
    usable = inspect_bank(bank, cfg, output_dir)
    all_folds = make_fold_plans(usable, cfg, output_dir)
    folds = select_folds(all_folds, args)
    write_scope(cfg, output_dir, [f["fold_id"] for f in folds])
    groups = fixed_groups(cfg)
    floor = configured_std_floor(cfg)
    budgets = resolve_budgets(cfg)

    print("=" * 78, flush=True)
    print(f"Device            : {device} ({gpu_name()})", flush=True)
    print(f"Output            : {output_dir}", flush=True)
    print(f"Mirror            : {args.mirror_dir or '(none)'}", flush=True)
    print(f"Split level       : {cfg['split_level']}", flush=True)
    print(f"All folds         : {len(all_folds)}; this session: "
          f"{[f['fold_id'] for f in folds]}", flush=True)
    print(f"seeds             : {cfg['seeds']}", flush=True)
    print(f"arms ({len(cfg['arms'])})         : {cfg['arms']}", flush=True)
    print(f"co-train groups   : {[len(g) for g in groups]} -> {groups}",
          flush=True)
    print(f"tile budgets      : {budgets} (None means use all)", flush=True)
    print(f"std floor         : {floor:g} (blank-plane error amplified "
          f"{1.0/floor:.3g}x)", flush=True)
    print(f"cache check       : mode={args.cache_check} "
          f"probe_chunks={args.cache_probe_chunks}", flush=True)
    print(f"signature         : {signature}", flush=True)
    print(memory_message(device), flush=True)
    print("Pause: press Ctrl+C once, or create a file named PAUSE in the "
          "output directory.", flush=True)
    print("=" * 78, flush=True)

    write_run_index(cfg, folds, output_dir, signature, all_folds)
    refresh_report(cfg, output_dir, args.mirror_dir, args.mirror_skip_weights)

    if args.plan_only:
        print("[PLAN ONLY] Nothing was trained.", flush=True)
        return

    session_pending = any(
        not completed_run(run_directory(output_dir, arm, fold["fold_id"], seed),
                          signature, arm_signature(cfg, arm))
        for fold in folds for seed in cfg["seeds"] for arm in cfg["arms"])
    if session_pending:
        audit_parameters(cfg, output_dir)

    session_log = []

    for fold in folds:
        stop.check()
        pending = any(
            not completed_run(run_directory(output_dir, arm, fold["fold_id"], seed),
                              signature, arm_signature(cfg, arm))
            for seed in cfg["seeds"] for arm in cfg["arms"])
        if not pending:
            print(f"[SKIP FOLD] {fold['fold_id']} already complete.",
                  flush=True)
            continue

        if stop.limit and stop.time_left() < 0.12 * stop.limit:
            print(f"[TIME LIMIT] Not enough time left to start "
                  f"{fold['fold_id']}; exiting cleanly.", flush=True)
            raise PauseRequested()

        fold_start = time.monotonic()
        print(f"\n[FOLD] {fold['fold_id']} test={fold['test_group']} "
              f"({len(fold['test_slides'])} slide(s), "
              f"{len(fold['packs']['test'])} tiles) "
              f"validation={fold['val_group']} "
              f"({len(fold['packs']['validation'])} tiles) "
              f"train={len(fold['packs']['train'])} tiles", flush=True)

        pipe = make_pipe(bank, cfg, device)
        stores = {}
        training_data = None
        cache_dir = Path(output_dir) / "eval_cache" / fold["fold_id"]
        qc_dir = Path(output_dir) / "qc" / fold["fold_id"]
        qc_dir.mkdir(parents=True, exist_ok=True)

        try:
            if cfg["cache_eval_preprocess"]:
                for split in ("validation", "test"):
                    stores[split] = build_eval_store(
                        bank, fold["packs"][split], pipe, cfg, cache_dir, split,
                        signature, stop, stats_dir=qc_dir,
                        rebuild=bool(args.rebuild_eval_cache))
                if cfg["validate_eval_cache"]:
                    for split in ("validation", "test"):
                        check_eval_store(
                            bank, fold["packs"][split], pipe, stores[split], cfg,
                            device, stop, tag=split, report_dir=qc_dir,
                            mode=args.cache_check,
                            probe_chunks=args.cache_probe_chunks,
                            rel_tol=args.cache_rel_tol,
                            corr_tol=args.cache_corr_tol)
                if want_val_cache_on_gpu(cfg, device, stores.get("validation")):
                    stores["validation"].to_device_cache(device)

            training_data = choose_training_data(bank, fold["packs"]["train"],
                                                 cfg, stop)

            for seed in cfg["seeds"]:
                for group_index, group in enumerate(groups):
                    stop.check()
                    train_group(group, group_index, fold, seed, bank,
                                training_data, pipe, stores, cfg, device,
                                output_dir, signature, stop,
                                args.save_every_batches, args.save_every_seconds)
                write_run_index(cfg, folds, output_dir, signature, all_folds)
                refresh_report(cfg, output_dir, args.mirror_dir,
                               args.mirror_skip_weights)
        finally:
            for store in stores.values():
                store.close()
            stores.clear()
            training_data = None
            release_device(device)

        minutes = (time.monotonic() - fold_start) / 60.0
        session_log.append({
            "fold_id": fold["fold_id"], "finished_at": utc_now(),
            "wall_minutes": round(minutes, 2),
            "seeds": ";".join(map(str, cfg["seeds"])),
            "n_arms": len(cfg["arms"]),
            "train_tiles": len(fold["packs"]["train"]),
            "val_tiles": len(fold["packs"]["validation"]),
            "test_tiles": len(fold["packs"]["test"]),
            "gpu": gpu_name(),
        })
        S.atomic_csv(pd.DataFrame(session_log),
                     Path(output_dir) / "session_timing.csv")
        print(f"[FOLD DONE] {fold['fold_id']} took {minutes:.1f} min",
              flush=True)

        if cfg["delete_eval_cache_after_fold"] and all(
                completed_run(run_directory(output_dir, arm, fold["fold_id"], seed),
                              signature, arm_signature(cfg, arm))
                for seed in cfg["seeds"] for arm in cfg["arms"]):
            for split in ("validation", "test"):
                for suffix in (".npy", ".json"):
                    path = cache_dir / f"{split}{suffix}"
                    if path.exists():
                        path.unlink()

    write_run_index(cfg, folds, output_dir, signature, all_folds)
    refresh_report(cfg, output_dir, args.mirror_dir, args.mirror_skip_weights)
    print("\n[DONE] All training assigned to this session is complete.",
          flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resumable multi-stain fusion benchmark trainer "
                    "(Colab-optimised)")
    parser.add_argument("--bank-dir", default=T.PATHS["bank_dir"])
    parser.add_argument("--output-base", default=T.PATHS["output_dir"])
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"],
                        default="auto")
    parser.add_argument("--run-name",
                        help="Fixed output directory name (strongly "
                             "recommended when merging multiple sessions)")
    parser.add_argument("--mirror-dir",
                        help="Incrementally mirror results here (e.g. a Google "
                             "Drive folder)")
    parser.add_argument("--mirror-skip-weights", action="store_true",
                        help="Skip inference.pt when mirroring (saves Drive "
                             "space)")
    parser.add_argument("--fold-ids", nargs="+",
                        help="Run only these folds (fold-000 or index 0)")
    parser.add_argument("--fold-shard",
                        help="Split all folds into N shards and run only the "
                             "i-th, written as i/N")
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="Save at a safe point and exit near this limit")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-tiles", help="integer or all")
    parser.add_argument("--val-tiles", help="integer or all")
    parser.add_argument("--test-tiles", help="integer or all")
    parser.add_argument("--max-tiles", type=int,
                        help="Legacy total budget (used only when no --*-tiles "
                             "are given)")
    parser.add_argument("--per-cell-std-floor", type=float)
    parser.add_argument("--no-eval-cache", action="store_true")
    parser.add_argument("--no-train-cache", action="store_true")
    parser.add_argument("--val-cache-gpu", choices=["auto", "on", "off"])
    parser.add_argument("--save-every-batches", type=int, default=100000)
    parser.add_argument("--save-every-seconds", type=float, default=1200.0)
    parser.add_argument("--pause-after-batches", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--baseline-arm", default=None)
    parser.add_argument("--cache-check", choices=list(CACHE_CHECK_MODES),
                        default="strict")
    parser.add_argument("--cache-probe-chunks", type=int, default=1)
    parser.add_argument("--cache-rel-tol", type=float, default=None)
    parser.add_argument("--cache-corr-tol", type=float, default=None)
    parser.add_argument("--rebuild-eval-cache", action="store_true")
    parser.add_argument("--no-plane-stats", action="store_true")
    parser.add_argument("--resume-dir")
    parser.add_argument("--allow-environment-change", action="store_true")
    return parser.parse_args()


def _tile_arg(value):
    if value is None:
        return _MISSING
    text = str(value).strip().lower()
    if text in ("all", "none", "0", ""):
        return None
    return int(text)


def build_cfg(args):
    cfg = copy.deepcopy(CFG)

    if args.seeds is not None:
        cfg["seeds"] = list(args.seeds)
    if args.arms is not None:
        cfg["arms"] = list(args.arms)
    if args.group_size is not None:
        cfg["co_train_group_size"] = int(args.group_size)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.eval_batch_size is not None:
        cfg["eval_batch_size"] = int(args.eval_batch_size)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
        cfg["min_epochs"] = min(cfg["min_epochs"], cfg["epochs"])
        cfg["warmup_epochs"] = min(cfg["warmup_epochs"], cfg["epochs"] - 1)

    budgets = dict(cfg.get("split_tile_budgets") or {})
    for key, value in (("train", args.train_tiles),
                       ("validation", args.val_tiles),
                       ("test", args.test_tiles)):
        parsed = _tile_arg(value)
        if parsed is not _MISSING:
            budgets[key] = parsed
    if budgets:
        cfg["split_tile_budgets"] = budgets
    if args.max_tiles is not None:
        cfg["max_tiles_per_fold"] = int(args.max_tiles)
        if all(v is None for v in (args.train_tiles, args.val_tiles,
                                   args.test_tiles)) and not CFG.get(
                                       "split_tile_budgets"):
            cfg["split_tile_budgets"] = None

    if args.per_cell_std_floor is not None:
        cfg["per_cell_std_floor"] = float(args.per_cell_std_floor)
    if args.val_cache_gpu is not None:
        cfg["val_cache_on_gpu"] = args.val_cache_gpu
    if args.no_eval_cache:
        cfg["cache_eval_preprocess"] = False
    if args.no_train_cache:
        cfg["cache_training_tiles"] = False
    if args.no_plane_stats:
        cfg["eval_cache_plane_stats"] = False
    if args.no_excel:
        cfg["write_excel_report"] = False
    if args.baseline_arm:
        cfg["report_baseline_arm"] = args.baseline_arm

    return cfg


def main():
    args = parse_args()
    cfg = build_cfg(args)
    validate_cfg(cfg)

    if args.save_every_batches <= 0 or args.save_every_seconds <= 0:
        raise ValueError("Checkpoint interval must be positive.")
    if args.cache_probe_chunks < 1:
        raise ValueError("--cache-probe-chunks must be at least 1.")

    device = T.get_device(args.device)
    if args.device != "auto" and device.type != args.device:
        raise RuntimeError(f"Requested device is unavailable: {args.device}")

    bank_dir = Path(args.bank_dir).expanduser().resolve()
    base = Path(args.output_base).expanduser().resolve()
    if base == bank_dir:
        raise ValueError("The output directory must not be the bank "
                         "directory.")

    T.set_seed(cfg["global_seed"], cfg["strict_determinism"])
    configure_backends(device)

    bank = T.TileBank(str(bank_dir), preload=False)
    specification = recipe_specification(cfg, bank, device)
    output_dir, signature, info = resolve_output_dir(base, specification, cfg, args)
    S.atomic_json(dataset_detail(bank), output_dir / "dataset_detail.json")

    with exclusive_lock(output_dir / ".writer.lock",
                        "Another process is already writing this output "
                        "directory."):
        stop = StopController(output_dir, args.pause_after_batches, args.max_hours)
        try:
            execute(cfg, bank, device, signature, output_dir, args, stop)
        except PauseRequested:
            stop.acknowledge()
            refresh_report(cfg, output_dir, args.mirror_dir,
                           args.mirror_skip_weights)
            print("\n[PAUSED] Exited safely. Run the same command again to "
                  "resume. Unfinished evaluation caches are rebuilt; committed "
                  "training progress is kept.", flush=True)
        except Exception:
            print("\n[ERROR] Training interrupted. No unsafe half-batch "
                  "snapshot was written; the last committed checkpoint is "
                  "kept.", flush=True)
            if info.get("mode") == "adopted":
                print("[NOTE] This directory was adopted under a stored "
                      "signature; completed runs remain valid.", flush=True)
            try:
                refresh_report(cfg, output_dir, args.mirror_dir,
                               args.mirror_skip_weights)
            except Exception:
                pass
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()

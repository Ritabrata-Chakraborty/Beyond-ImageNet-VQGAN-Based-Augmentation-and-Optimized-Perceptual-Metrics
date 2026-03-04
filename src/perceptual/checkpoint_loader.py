"""
Load backbone features from train_backbone, LPIPS, or weights/v0.1 checkpoints for layer selection.
"""

from __future__ import annotations

import os
import re
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PERCEPTUAL_SIMILARITY = os.path.join(_PROJECT_ROOT, "external", "perceptual_similarity")

# Standard layer indices (must match layer_selection.get_layer_indices_and_names)
_STANDARD_INDICES = {
    "vgg16": [1, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 27, 30],
    "alexnet": [1, 4, 7, 9, 11],
    "squeezenet": [2, 3, 5, 6, 8, 9, 10, 11, 12],
}


def _remap_lpips_slice_keys_to_features(
    state: dict,
    net_type: str,
    layer_indices: list[int],
) -> dict:
    """Remap net.sliceN.M.* -> net.features.M.* for loading into features-only model."""
    if _PERCEPTUAL_SIMILARITY not in sys.path:
        sys.path.insert(0, _PERCEPTUAL_SIMILARITY)
    import lpips.pretrained_networks as pn

    if not layer_indices:
        return state
    if net_type in ("vgg", "vgg16"):
        endpoints = [pn.VGG16_LAYER_ENDPOINTS[i] for i in layer_indices]
    elif net_type == "alexnet":
        endpoints = [pn.ALEXNET_LAYER_ENDPOINTS[i] for i in layer_indices]
    elif net_type == "squeezenet":
        endpoints = [pn.SQUEEZENET_LAYER_ENDPOINTS[i] for i in layer_indices]
    else:
        return state
    starts = [0] + endpoints[:-1]
    pattern = re.compile(r"^net\.slice(\d+)\.(\d+)\.(.+)$")
    new_state = dict(state)
    for key in list(new_state.keys()):
        m = pattern.match(key)
        if m is None:
            continue
        n_slice, m_mod, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
        if n_slice < 1 or n_slice > len(starts):
            continue
        if not (starts[n_slice - 1] <= m_mod < endpoints[n_slice - 1]):
            continue
        new_key = f"net.features.{m_mod}.{suffix}"
        if new_key not in new_state:
            new_state[new_key] = new_state[key]
            del new_state[key]
    return new_state


def _max_feature_index_from_state(state: dict) -> int:
    """Return max feature index present in keys net.features.<i>.* (-1 if none)."""
    pattern = re.compile(r"^net\.features\.(\d+)\.(.+)$")
    max_idx = -1
    for key in state:
        m = pattern.match(key)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


class _FeaturesWrapper(nn.Module):
    """Thin wrapper so layer_selection can use get_layer_indices_and_names(wrapper, model_name)."""

    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features


def load_backbone_for_features(
    checkpoint_path: str,
    model_name: str,
    checkpoint_type: str,
    lpips_layers: list[int] | None = None,
) -> tuple[nn.Module, list[int]]:
    """Load backbone features and return (wrapper with .features, available layer indices).

    checkpoint_type: "train_backbone" | "lpips" | "weights_v01"
    model_name: vgg16 | alexnet | squeezenet
    lpips_layers: for lpips/weights_v01, layer indices used when training (e.g. [0,2,4,7,10]).
    """
    if checkpoint_type not in ("train_backbone", "lpips", "weights_v01"):
        raise ValueError(f"checkpoint_type must be train_backbone, lpips, or weights_v01, got {checkpoint_type}")
    if model_name not in ("vgg16", "alexnet", "squeezenet"):
        raise ValueError(f"model_name must be vgg16, alexnet, or squeezenet, got {model_name}")

    if checkpoint_type == "train_backbone":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_backbone import build_model

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" not in ckpt:
            raise ValueError(f"train_backbone checkpoint must contain model_state_dict: {checkpoint_path}")
        model = build_model(model_name, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        indices = _STANDARD_INDICES[model_name]
        return _FeaturesWrapper(model.features), indices

    # LPIPS or weights_v01
    if _PERCEPTUAL_SIMILARITY not in sys.path:
        sys.path.insert(0, _PERCEPTUAL_SIMILARITY)
    import lpips.pretrained_networks as pn

    net_map = {"vgg16": "vgg16", "alexnet": "alexnet", "squeezenet": "squeezenet"}
    pn_net = net_map[model_name]
    if pn_net == "vgg16":
        default_layers = pn.VGG16_DEFAULT_LAYERS
    elif pn_net == "alexnet":
        default_layers = pn.ALEXNET_DEFAULT_LAYERS
    else:
        default_layers = pn.SQUEEZENET_DEFAULT_LAYERS

    layers = lpips_layers if lpips_layers is not None else default_layers
    if pn_net == "alexnet" and layers is None:
        layers = list(range(len(pn.ALEXNET_DEFAULT_ENDPOINTS)))

    if pn_net == "vgg16":
        backbone = pn.vgg16(pretrained=False, layer_indices=layers)
    elif pn_net == "alexnet":
        backbone = pn.alexnet(pretrained=False, layer_indices=layers)
    else:
        backbone = pn.squeezenet(pretrained=False, layer_indices=layers)

    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"LPIPS/weights checkpoint must be a state dict: {checkpoint_path}")

    state = _remap_lpips_slice_keys_to_features(state, pn_net, layers)
    max_idx = _max_feature_index_from_state(state)

    wrapper_net = nn.Module()
    wrapper_net.net = backbone
    wrapper_net.load_state_dict(state, strict=False)

    full_indices = _STANDARD_INDICES[model_name]
    if max_idx >= 0:
        available_indices = [i for i in full_indices if i <= max_idx]
    else:
        available_indices = full_indices

    return _FeaturesWrapper(backbone.features), available_indices

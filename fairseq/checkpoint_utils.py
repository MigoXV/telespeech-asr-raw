import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqDecoder, FairseqEncoder

logger = logging.getLogger(__name__)


def _apply_arg_overrides(state: Dict[str, Any], arg_overrides: Optional[Dict[str, Any]]):
    if not arg_overrides:
        return state

    if "cfg" in state and state["cfg"] is not None:
        cfg = OmegaConf.create(state["cfg"])
        with open_dict(cfg):
            for k, v in arg_overrides.items():
                OmegaConf.update(cfg, k, v, force_add=True)
        state["cfg"] = cfg
    elif "args" in state and state["args"] is not None:
        for k, v in arg_overrides.items():
            setattr(state["args"], k, v)
    return state


def load_checkpoint_to_cpu(path: str, arg_overrides: Optional[Dict[str, Any]] = None):
    state = torch.load(path, map_location="cpu")
    return _apply_arg_overrides(state, arg_overrides)


def _build_cfg_from_state(state: Dict[str, Any]):
    if "cfg" in state and state["cfg"] is not None:
        return state["cfg"]
    if "args" in state and state["args"] is not None:
        return convert_namespace_to_omegaconf(state["args"])
    raise RuntimeError(f"No configuration found in checkpoint keys: {state.keys()}")


def load_model_ensemble(
    filenames: List[str],
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict: bool = True,
    suffix: str = "",
    num_shards: int = 1,
    state: Optional[Dict[str, Any]] = None,
):
    ensemble, cfg, _task = load_model_ensemble_and_task(
        filenames,
        arg_overrides=arg_overrides,
        task=task,
        strict=strict,
        suffix=suffix,
        num_shards=num_shards,
        state=state,
    )
    return ensemble, cfg


def load_model_ensemble_and_task(
    filenames: List[str],
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict: bool = True,
    suffix: str = "",
    num_shards: int = 1,
    state: Optional[Dict[str, Any]] = None,
):
    from fairseq import tasks

    assert num_shards == 1, "Checkpoint sharding is not supported without distributed support"

    ensemble: List[Any] = []
    cfg = None
    for filename in filenames:
        if not Path(filename).exists():
            raise IOError(f"Model file not found: {filename}")

        if state is None:
            state = load_checkpoint_to_cpu(filename, arg_overrides)

        cfg = _build_cfg_from_state(state)

        if task is None:
            task = tasks.setup_task(cfg.task)

        if "task_state" in state:
            task.load_state_dict(state["task_state"])

        model = task.build_model(cfg.model)
        if (
            "optimizer_history" in state
            and state["optimizer_history"]
            and "num_updates" in state["optimizer_history"][-1]
        ):
            model.set_num_updates(state["optimizer_history"][-1]["num_updates"])

        model_state = prune_state_dict(state.get("model", {}), cfg.model)
        model.load_state_dict(model_state, strict=strict)
        ensemble.append(model)

        state = None

    return ensemble, cfg, task


def load_pretrained_component_from_model(
    component: Any,
    checkpoint: str,
    strict: bool = True,
    state_prefix: Optional[str] = None,
):
    state = load_checkpoint_to_cpu(checkpoint)
    model_state = state.get("model", {})

    if state_prefix is not None:
        filtered_state = {}
        for k, v in model_state.items():
            if k.startswith(state_prefix + "."):
                filtered_state[k[len(state_prefix) + 1 :]] = v
        model_state = filtered_state

    missing, unexpected = component.load_state_dict(model_state, strict=strict)
    if missing:
        logger.warning("Missing keys when loading pretrained component: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading pretrained component: %s", unexpected)
    return component


def prune_state_dict(state_dict: Dict[str, Any], model_cfg: Optional[DictConfig]):
    arch = None
    if model_cfg is not None:
        arch = (
            model_cfg._name
            if isinstance(model_cfg, DictConfig)
            else getattr(model_cfg, "arch", None)
        )

    if not model_cfg or arch is None or arch == "ptt_transformer":
        return state_dict

    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    logger.info(
        "Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(int(layer_string) for layer_string in layers_to_keep.split(","))
        mapping_dict = {str(keep_layers[i]): str(i) for i in range(len(keep_layers))}
        regex = re.compile(r"^{layer}.*\.layers\.(\d+)".format(layer=layer_name))
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = state_dict.copy()
    for key in list(new_state_dict.keys()):
        for pruning_pass in pruning_passes:
            regex = pruning_pass["substitution_regex"]
            result = re.search(regex, key)
            if result is None:
                continue

            current_layer_number = result.group(1)
            if current_layer_number not in pruning_pass["mapping_dict"]:
                del new_state_dict[key]
                continue

            new_layer_number = pruning_pass["mapping_dict"][current_layer_number]
            new_key = re.sub(regex, f"{result.group(0)[: result.start(1)]}{new_layer_number}", key)
            new_state_dict[new_key] = new_state_dict.pop(key)
            break

    return new_state_dict


def load_ema_from_checkpoint(fpath: str):
    state = torch.load(fpath, map_location="cpu")
    if "extra_state" in state and "ema" in state["extra_state"]:
        return state["extra_state"]["ema"]
    if "ema" in state:
        return state["ema"]
    raise RuntimeError(f"EMA weights not found in checkpoint: {fpath}")


def load_checkpoint_to_cpu_with_maybe_component(
    component: Optional[Any], src_path: str, map_location=None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state = torch.load(src_path, map_location=map_location or "cpu")
    if component is None:
        return state, state.get("model", {})

    component_state: Dict[str, Any] = {}
    for k, v in state.get("model", {}).items():
        if isinstance(component, FairseqEncoder) and k.startswith("encoder."):
            component_state[k.replace("encoder.", "")] = v
        elif isinstance(component, FairseqDecoder) and k.startswith("decoder."):
            component_state[k.replace("decoder.", "")] = v
    return state, component_state


def load_ema_checkpoint_to_cpu(src_path: str):
    state = load_ema_from_checkpoint(src_path)
    if "state_dict" in state:
        state = state["state_dict"]
    return state

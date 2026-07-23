"""
Export M-LSD TensorFlow checkpoints to split head/tail ONNX models for split computing.

Edit SplitExportConfig below, then run:
    python onnx/export_onnx_split.py

Optional CLI overrides: --checkpoint-dir, --output-dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow.keras.applications import mobilenet_v2

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
fine_tuning_dir = os.path.join(repo_root, "fine-tuning")
sys.path.insert(0, repo_root)
sys.path.insert(0, fine_tuning_dir)

from load_model import ModelConfig, build_model, infer_config_from_path, load_checkpoint


# ---------------------------------------------------------------------------
# Configuration (edit these values)
# ---------------------------------------------------------------------------

SPLIT_POINT_CHOICES = ("block_1_project", "block_3_project", "block_6_project")

# MobileNetV2 layer indices inside the extractor sub-model (TF 2.4+ pick_layer - 1).
PICK_LAYER_INDICES: Dict[str, Dict[str, int]] = {
    "MLSD": {"p1": 26, "p2": 53, "p3": 89},
    "MLSD_large": {"p0": 8, "p1": 26, "p2": 53, "p3": 89, "p4": 115},
}


@dataclass
class SplitExportConfig:
    input_size: int = 512
    backbone_type: str = "MLSD"  # MLSD=tiny, MLSD_large=large
    split_point: str = "block_1_project"
    checkpoint_dir: str = "fine_tuned_model/M-LSD_512_tiny_ft_100_00005"
    output_dir: str = "fine_tuned_model/onnx/split"
    batch_size: int = 1
    topk: int = 200
    onnx_opset: int = 11
    infer_from_ckpt: bool = True
    run_validation: bool = True
    validation_tolerance: float = 1e-4

    def variant_label(self) -> str:
        return "tiny" if self.backbone_type == "MLSD" else "large"

    def output_stem(self) -> str:
        return f"M-LSD_{self.input_size}_{self.variant_label()}_split_{self.split_point}"

    def to_model_config(self, ckpt_dir: str) -> ModelConfig:
        if self.infer_from_ckpt:
            try:
                inferred = infer_config_from_path(ckpt_dir)
                return ModelConfig(
                    input_size=inferred.input_size,
                    map_size=inferred.map_size,
                    backbone_type=inferred.backbone_type,
                    batch_size=self.batch_size,
                    topk=self.topk,
                )
            except ValueError:
                pass
        return ModelConfig(
            input_size=self.input_size,
            map_size=self.input_size // 2,
            backbone_type=self.backbone_type,
            batch_size=self.batch_size,
            topk=self.topk,
        )


@dataclass(frozen=True)
class SplitSpec:
    split_point: str
    split_layer_suffix: str
    head_tap_ids: Tuple[str, ...]
    tail_tap_ids: Tuple[str, ...]
    pick_indices: Dict[str, int]
    split_layer_index: int


# ---------------------------------------------------------------------------
# Checkpoint / export helpers
# ---------------------------------------------------------------------------


def resolve_checkpoint_dir(path: str) -> str:
    if os.path.isfile(path):
        raise ValueError(f"Checkpoint path must be a directory, got file: {path}")
    if os.path.exists(os.path.join(path, "checkpoint")):
        return path
    if not os.path.isdir(path):
        raise ValueError(f"Checkpoint path does not exist: {path}")
    candidates = []
    for name in os.listdir(path):
        subdir = os.path.join(path, name)
        if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "checkpoint")):
            candidates.append(subdir)
    if not candidates:
        raise ValueError(f"No checkpoint directory found under: {path}")
    return max(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "checkpoint")))


def build_export_model(model: tf.keras.Model) -> tf.keras.Model:
    def passthrough(x):
        return x

    outputs = [
        tf.keras.layers.Lambda(passthrough, name="org_disp_map")(model.output[10]),
        tf.keras.layers.Lambda(passthrough, name="org_center_pts")(model.output[11]),
        tf.keras.layers.Lambda(passthrough, name="org_center_scores")(model.output[12]),
    ]
    return tf.keras.Model(model.input, outputs, name="WireFrameModel")


def _extractor_name(cfg: ModelConfig) -> str:
    return f"{cfg.backbone_type}{cfg.post_name}"


def _get_extractor(full_model: tf.keras.Model, cfg: ModelConfig) -> tf.keras.Model:
    return full_model.get_layer(_extractor_name(cfg))


def _get_submodule(full_model: tf.keras.Model, name_prefix: str) -> tf.keras.layers.Layer:
    for layer in full_model.layers:
        if layer.name == name_prefix or layer.name.startswith(f"{name_prefix}_"):
            return layer
    raise ValueError(f"No layer matching prefix {name_prefix!r}. Found: {[l.name for l in full_model.layers]}")


def _find_split_layer_index(layers: Sequence[tf.keras.layers.Layer], split_layer_suffix: str) -> int:
    bn_name = f"{split_layer_suffix}_BN"
    for idx, layer in enumerate(layers):
        if layer.name == bn_name or layer.name.endswith(bn_name):
            return idx
    for idx, layer in enumerate(layers):
        if split_layer_suffix in layer.name and "BN" not in layer.name:
            return idx
    raise ValueError(f"Split layer not found: {split_layer_suffix}")


def build_split_spec(cfg: ModelConfig, split_point: str) -> SplitSpec:
    if split_point not in SPLIT_POINT_CHOICES:
        raise ValueError(f"Unsupported split_point={split_point!r}. Choose from {SPLIT_POINT_CHOICES}")

    pick_indices = PICK_LAYER_INDICES[cfg.backbone_type]
    model = build_model(cfg)
    extractor = _get_extractor(model, cfg)
    split_layer_index = _find_split_layer_index(extractor.layers, split_point)

    head_tap_ids: List[str] = []
    tail_tap_ids: List[str] = []
    for tap_id, layer_idx in sorted(pick_indices.items(), key=lambda kv: kv[1]):
        if layer_idx <= split_layer_index:
            head_tap_ids.append(tap_id)
        else:
            tail_tap_ids.append(tap_id)

    return SplitSpec(
        split_point=split_point,
        split_layer_suffix=split_point,
        head_tap_ids=tuple(head_tap_ids),
        tail_tap_ids=tuple(tail_tap_ids),
        pick_indices=pick_indices,
        split_layer_index=split_layer_index,
    )


def _tap_output_name(tap_id: str) -> str:
    return f"tap_{tap_id}"


# ---------------------------------------------------------------------------
# Head / tail model builders
# ---------------------------------------------------------------------------


def _forward_backbone_layers(
    layers: Sequence[tf.keras.layers.Layer],
    x: tf.Tensor,
    *,
    start_idx: int,
    end_idx: int,
    pick_indices: Dict[str, int],
    collect_tap_ids: Sequence[str],
    training: bool = False,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    taps: Dict[str, tf.Tensor] = {}
    block_input = x

    for layer_idx, layer in enumerate(layers):
        if layer_idx < start_idx:
            continue
        if layer_idx > end_idx:
            break

        layer_name = layer.name
        if (
            "_expand" in layer_name
            and "relu" not in layer_name
            and "BN" not in layer_name
            and layer.__class__.__name__ == "Conv2D"
        ):
            block_input = x

        if isinstance(layer, tf.keras.layers.Add):
            x = layer([x, block_input], training=training)
        else:
            x = layer(x, training=training)

        for tap_id in collect_tap_ids:
            if layer_idx == pick_indices[tap_id]:
                taps[tap_id] = x

    return x, taps


def _continue_backbone_layers(
    layers: Sequence[tf.keras.layers.Layer],
    x: tf.Tensor,
    *,
    start_idx: int,
    pick_indices: Dict[str, int],
    collect_tap_ids: Sequence[str],
    training: bool = False,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    taps: Dict[str, tf.Tensor] = {}
    block_input = x

    for layer_idx, layer in enumerate(layers):
        if layer_idx <= start_idx:
            continue

        layer_name = layer.name
        if (
            "_expand" in layer_name
            and "relu" not in layer_name
            and "BN" not in layer_name
            and layer.__class__.__name__ == "Conv2D"
        ):
            block_input = x

        if isinstance(layer, tf.keras.layers.Add):
            x = layer([x, block_input], training=training)
        else:
            x = layer(x, training=training)

        for tap_id in collect_tap_ids:
            if layer_idx == pick_indices[tap_id]:
                taps[tap_id] = x

    return x, taps


class SplitTailModel(tf.keras.Model):
    """Server-side model: tap inputs + backbone continuation -> 3 export outputs."""

    def __init__(
        self,
        backbone_layers: Sequence[tf.keras.layers.Layer],
        split_spec: SplitSpec,
        decoder_fpn: tf.keras.layers.Layer,
        decoder: tf.keras.layers.Layer,
        head_tap_ids: Sequence[str],
        input_shapes: Dict[str, Tuple[int, ...]],
        batch_size: int,
    ):
        super().__init__(name="split_tail")
        self.backbone_layers = list(backbone_layers)
        self.split_spec = split_spec
        self.decoder_fpn = decoder_fpn
        self.decoder = decoder
        self.head_tap_ids = list(head_tap_ids)
        self.tail_tap_ids = list(split_spec.tail_tap_ids)
        self.pick_indices = split_spec.pick_indices
        self.split_layer_index = split_spec.split_layer_index
        self.batch_size = batch_size

        self._tap_inputs: Dict[str, tf.keras.layers.Input] = {}
        for tap_id in self.head_tap_ids:
            self._tap_inputs[tap_id] = tf.keras.layers.Input(
                shape=input_shapes[_tap_output_name(tap_id)],
                dtype=tf.float32,
                name=_tap_output_name(tap_id),
            )

        self._split_input = tf.keras.layers.Input(
            shape=input_shapes["split_state"],
            dtype=tf.float32,
            name="split_state",
        )

    def call(self, inputs=None, training=False):
        if inputs is None:
            ordered_head = [self._tap_inputs[tap_id] for tap_id in self.head_tap_ids]
            split_state = self._split_input
        elif isinstance(inputs, dict):
            ordered_head = [inputs[_tap_output_name(tap_id)] for tap_id in self.head_tap_ids]
            split_state = inputs["split_state"]
        else:
            n_head = len(self.head_tap_ids)
            ordered_head = list(inputs[:n_head])
            split_state = inputs[n_head]

        tap_values: Dict[str, tf.Tensor] = {}
        for tap_id, tensor in zip(self.head_tap_ids, ordered_head):
            tap_values[tap_id] = tensor

        _, tail_taps = _continue_backbone_layers(
            self.backbone_layers,
            split_state,
            start_idx=self.split_layer_index,
            pick_indices=self.pick_indices,
            collect_tap_ids=self.tail_tap_ids,
            training=training,
        )
        tap_values.update(tail_taps)

        ordered_tap_ids = sorted(self.pick_indices.keys(), key=lambda k: self.pick_indices[k])
        x_list = [tap_values[tap_id] for tap_id in ordered_tap_ids]

        fpn_out = self.decoder_fpn(x_list)
        decoder_out = self.decoder(fpn_out, training=training)
        return decoder_out[10], decoder_out[11], decoder_out[12]


def build_head_model(
    full_model: tf.keras.Model,
    cfg: ModelConfig,
    split_spec: SplitSpec,
) -> tf.keras.Model:
    extractor = _get_extractor(full_model, cfg)
    layers = extractor.layers

    inp = tf.keras.layers.Input(
        shape=(cfg.input_size, cfg.input_size, 3),
        name="input_image",
    )
    x = mobilenet_v2.preprocess_input(inp)

    split_state, head_tensors = _forward_backbone_layers(
        layers,
        x,
        start_idx=0,
        end_idx=split_spec.split_layer_index,
        pick_indices=split_spec.pick_indices,
        collect_tap_ids=split_spec.head_tap_ids,
        training=False,
    )

    output_tensors = []
    for tap_id in split_spec.head_tap_ids:
        output_tensors.append(
            tf.keras.layers.Lambda(
                lambda t, tap_id=tap_id: t,
                name=_tap_output_name(tap_id),
            )(head_tensors[tap_id])
        )
    split_out = tf.keras.layers.Lambda(lambda t: t, name="split_state")(split_state)
    output_tensors.append(split_out)
    return tf.keras.Model(inp, output_tensors, name="split_head")


def head_output_names(split_spec: SplitSpec) -> List[str]:
    return [_tap_output_name(tap_id) for tap_id in split_spec.head_tap_ids] + ["split_state"]


def _concrete_head_output_shapes(
    head_model: tf.keras.Model,
    output_names: Sequence[str],
    batch_size: int,
    input_size: int,
) -> Dict[str, Tuple[int, ...]]:
    dummy = tf.zeros((batch_size, input_size, input_size, 3), dtype=tf.float32)
    outputs = head_model(dummy, training=False)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    shapes: Dict[str, Tuple[int, ...]] = {}
    for name, value in zip(output_names, outputs):
        shape = tuple(int(d) for d in value.shape)
        if len(shape) == 3:
            shape = (batch_size,) + shape
        shapes[name] = shape
    return shapes


def build_tail_model(
    full_model: tf.keras.Model,
    cfg: ModelConfig,
    split_spec: SplitSpec,
    head_model: tf.keras.Model,
    head_output_shapes: Dict[str, Tuple[int, ...]],
) -> SplitTailModel:
    input_shapes: Dict[str, Tuple[int, ...]] = {
        name: shape[1:] for name, shape in head_output_shapes.items()
    }

    tail = SplitTailModel(
        backbone_layers=_get_extractor(full_model, cfg).layers,
        split_spec=split_spec,
        decoder_fpn=_get_submodule(full_model, "decoder_fpn"),
        decoder=_get_submodule(full_model, "Decoder"),
        head_tap_ids=split_spec.head_tap_ids,
        input_shapes=input_shapes,
        batch_size=cfg.batch_size,
    )

    ordered_head = [tail._tap_inputs[tap_id] for tap_id in split_spec.head_tap_ids]
    tail_outputs = tail(ordered_head + [tail._split_input], training=False)
    tail.model = tf.keras.Model(
        inputs=ordered_head + [tail._split_input],
        outputs=list(tail_outputs),
        name="split_tail",
    )
    return tail


def _export_keras_to_onnx(model: tf.keras.Model, output_path: str, opset: int, batch_size: int = 1) -> None:
    inputs = model.input if isinstance(model.input, list) else [model.input]
    input_signature = []
    for inp in inputs:
        static_shape = [batch_size]
        for dim in inp.shape[1:]:
            static_shape.append(int(dim) if dim is not None else None)
        input_signature.append(
            tf.TensorSpec(static_shape, tf.float32, name=inp.name.split(":")[0])
        )

    tf2onnx.convert.from_keras(
        model,
        input_signature=tuple(input_signature),
        opset=opset,
        output_path=output_path,
    )


def _tensor_spec(shape: Tuple[int, ...]) -> Dict:
    return {"shape": list(int(d) for d in shape), "numel": int(np.prod(shape))}


def write_split_meta(
    path: str,
    cfg: ModelConfig,
    split_spec: SplitSpec,
    head_model: tf.keras.Model,
    tail_model: SplitTailModel,
    ckpt_dir: str,
    head_output_shapes: Dict[str, Tuple[int, ...]],
) -> None:
    input_numel = cfg.batch_size * cfg.input_size * cfg.input_size * 3
    head_names = head_output_names(split_spec)
    head_out_specs = [_tensor_spec(head_output_shapes[name]) for name in head_names]
    head_total_numel = sum(s["numel"] for s in head_out_specs)

    meta = {
        "checkpoint": ckpt_dir,
        "input_size": cfg.input_size,
        "map_size": cfg.map_size,
        "backbone_type": cfg.backbone_type,
        "variant": "tiny" if cfg.backbone_type == "MLSD" else "large",
        "split_point": split_spec.split_point,
        "split_layer_index": split_spec.split_layer_index,
        "head_tap_ids": list(split_spec.head_tap_ids),
        "tail_tap_ids": list(split_spec.tail_tap_ids),
        "pick_layer_indices": split_spec.pick_indices,
        "input_numel": input_numel,
        "head_outputs": {
            name: spec for name, spec in zip(head_names, head_out_specs)
        },
        "head_total_numel": head_total_numel,
        "compression_ratio_vs_input": head_total_numel / input_numel,
        "tail_outputs": ["org_disp_map", "org_center_pts", "org_center_scores"],
        "tail_input_names": [inp.name.split(":")[0] for inp in tail_model.model.inputs],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def validate_split_models(
    full_export_model: tf.keras.Model,
    head_model: tf.keras.Model,
    tail_model: SplitTailModel,
    split_spec: SplitSpec,
    cfg: ModelConfig,
    tolerance: float,
) -> None:
    rng = np.random.default_rng(0)
    x_np = rng.random((cfg.batch_size, cfg.input_size, cfg.input_size, 3), dtype=np.float32)
    x = tf.constant(x_np)

    full_out = full_export_model(x, training=False)
    head_out = head_model(x, training=False)
    if not isinstance(head_out, (list, tuple)):
        head_out = [head_out]
    tail_inputs = []
    for value in head_out:
        if len(value.shape) == 3:
            value = tf.expand_dims(value, axis=0)
        tail_inputs.append(value)
    tail_out = tail_model.model(tail_inputs, training=False)

    names = ["org_disp_map", "org_center_pts", "org_center_scores"]
    print("[validation]")
    ok = True
    for name, ref, got in zip(names, full_out, tail_out):
        diff = float(tf.reduce_max(tf.abs(ref - got)).numpy())
        passed = diff <= tolerance
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: max_abs_diff={diff:.6e} [{status}]")

    if not ok:
        raise RuntimeError("Split head/tail validation failed. See diffs above.")


def export_split_onnx(
    export_cfg: SplitExportConfig,
    ckpt_dir: str,
    output_dir: str,
) -> None:
    cfg = export_cfg.to_model_config(ckpt_dir)
    split_spec = build_split_spec(cfg, export_cfg.split_point)

    full_model = build_model(cfg)
    load_checkpoint(full_model, ckpt_dir)
    full_export_model = build_export_model(full_model)

    head_model = build_head_model(full_model, cfg, split_spec)
    head_names = head_output_names(split_spec)
    head_output_shapes = _concrete_head_output_shapes(
        head_model, head_names, cfg.batch_size, cfg.input_size
    )
    tail_wrapper = build_tail_model(full_model, cfg, split_spec, head_model, head_output_shapes)
    tail_model = tail_wrapper.model

    # Name tail outputs for ONNX consumers.
    org_disp, org_pts, org_scores = tail_model.outputs
    tail_named = tf.keras.Model(
        tail_model.inputs,
        [
            tf.keras.layers.Lambda(lambda t: t, name="org_disp_map")(org_disp),
            tf.keras.layers.Lambda(lambda t: t, name="org_center_pts")(org_pts),
            tf.keras.layers.Lambda(lambda t: t, name="org_center_scores")(org_scores),
        ],
        name="split_tail",
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = export_cfg.output_stem()
    head_path = os.path.join(output_dir, f"{stem}_head.onnx")
    tail_path = os.path.join(output_dir, f"{stem}_tail.onnx")
    meta_path = os.path.join(output_dir, f"{stem}_split_meta.json")

    print(f"[*] checkpoint: {ckpt_dir}")
    print(f"[*] split_point: {split_spec.split_point} (layer index {split_spec.split_layer_index})")
    print(f"[*] head taps: {split_spec.head_tap_ids} + split_state")
    print(f"[*] tail taps (computed on server): {split_spec.tail_tap_ids}")
    print(f"[*] input size: {cfg.input_size}, backbone: {cfg.backbone_type}, topk: {cfg.topk}")
    print(f"[*] opset: {export_cfg.onnx_opset}")

    if export_cfg.run_validation:
        validate_split_models(
            full_export_model,
            head_model,
            tail_wrapper,
            split_spec,
            cfg,
            export_cfg.validation_tolerance,
        )

    _export_keras_to_onnx(head_model, head_path, export_cfg.onnx_opset, cfg.batch_size)
    _export_keras_to_onnx(tail_named, tail_path, export_cfg.onnx_opset, cfg.batch_size)
    write_split_meta(meta_path, cfg, split_spec, head_model, tail_wrapper, ckpt_dir, head_output_shapes)

    print(f"[*] head: {head_path}")
    print(f"[*] tail: {tail_path}")
    print(f"[*] meta: {meta_path}")
    print("[*] export complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export split head/tail ONNX models for M-LSD.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override SplitExportConfig.checkpoint_dir",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override SplitExportConfig.output_dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_cfg = SplitExportConfig()

    ckpt_path = args.checkpoint_dir or export_cfg.checkpoint_dir
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(repo_root, ckpt_path)
    ckpt_dir = resolve_checkpoint_dir(ckpt_path)

    output_dir = args.output_dir or export_cfg.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(repo_root, output_dir)

    export_split_onnx(export_cfg, ckpt_dir, output_dir)


if __name__ == "__main__":
    main()

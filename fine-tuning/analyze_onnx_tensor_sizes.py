"""
Analyze ONNX graph tensor shapes/sizes and summarize split-cut candidates.

Finds nodes whose output tensors are smaller than the model input tensor
(useful for split-computing cut points), then prints a compact summary:
  - First node whose output numel < input numel
  - Representative nodes (first occurrence per first-output shape)
  - Smallest-output nodes

JSON outputs are written under onnx_analysis/ at the repository root.

Examples:
  python fine-tuning/analyze_onnx_tensor_sizes.py --model model.onnx
  python fine-tuning/analyze_onnx_tensor_sizes.py --model model.onnx --save-json custom_name.json
  python fine-tuning/analyze_onnx_tensor_sizes.py --json-path onnx_analysis/custom_name.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import onnx
from onnx import shape_inference

# All analysis JSON files go here (repo root / onnx_analysis).
DEFAULT_JSON_DIR = Path(__file__).resolve().parents[1] / "onnx_analysis"


def _dims_to_list(dims: Iterable[onnx.TensorShapeProto.Dimension]) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for d in dims:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        else:
            out.append(None)
    return out


def _numel(shape: Sequence[Optional[int]]) -> Optional[int]:
    if not shape:
        return 0
    if any(d is None for d in shape):
        return None
    vals = [int(d) for d in shape if d is not None]
    if any(v < 0 for v in vals):
        return None
    return int(reduce(mul, vals, 1))


def _is_initializer(model: onnx.ModelProto, name: str) -> bool:
    return any(init.name == name for init in model.graph.initializer)


def _collect_value_info(model: onnx.ModelProto) -> Dict[str, Tuple[int, List[Optional[int]]]]:
    """Map tensor name -> (elem_type, shape_dims) where shape dims are int or None."""
    out: Dict[str, Tuple[int, List[Optional[int]]]] = {}

    def add_vi(vi: onnx.ValueInfoProto) -> None:
        if not vi.type.HasField("tensor_type"):
            return
        tt = vi.type.tensor_type
        out[vi.name] = (int(tt.elem_type), _dims_to_list(tt.shape.dim))

    for vi in model.graph.input:
        add_vi(vi)
    for vi in model.graph.output:
        add_vi(vi)
    for vi in model.graph.value_info:
        add_vi(vi)

    for init in model.graph.initializer:
        out.setdefault(init.name, (int(init.data_type), [int(x) for x in init.dims]))

    return out


def _pick_model_input_tensor(model: onnx.ModelProto) -> onnx.ValueInfoProto:
    """Heuristic: pick the first graph input that is not an initializer."""
    for vi in model.graph.input:
        if not _is_initializer(model, vi.name):
            return vi
    return model.graph.input[0]


@dataclass(frozen=True)
class NodeSizeRow:
    index: int
    name: str
    op_type: str
    outputs: List[str]
    output_shapes: List[List[Optional[int]]]
    output_numels: List[Optional[int]]
    min_output_numel: Optional[int]
    smaller_than_input: Optional[bool]


def _row_to_dict(r: NodeSizeRow) -> Dict[str, Any]:
    return {
        "index": r.index,
        "name": r.name,
        "op_type": r.op_type,
        "outputs": r.outputs,
        "output_shapes": r.output_shapes,
        "output_numels": r.output_numels,
        "min_output_numel": r.min_output_numel,
        "smaller_than_input": r.smaller_than_input,
    }


def analyze(model_path: str) -> Dict[str, Any]:
    model = onnx.load(model_path)
    infer_err: Optional[str] = None
    try:
        inferred = shape_inference.infer_shapes(model)
        infer_ok = True
    except Exception as e:
        inferred = model
        infer_ok = False
        infer_err = str(e)

    vi = _collect_value_info(inferred)
    input_vi = _pick_model_input_tensor(inferred)
    _input_elem_type, input_shape = vi.get(input_vi.name, (None, []))  # type: ignore[assignment]
    input_numel = _numel(input_shape)

    rows: List[NodeSizeRow] = []
    for idx, node in enumerate(inferred.graph.node):
        out_names = list(node.output)
        out_shapes: List[List[Optional[int]]] = []
        out_numels: List[Optional[int]] = []
        for oname in out_names:
            shape = vi.get(oname, (None, []))[1]
            out_shapes.append(shape)
            out_numels.append(_numel(shape))

        known = [n for n in out_numels if n is not None]
        min_out = min(known) if known else None
        smaller = None
        if input_numel is not None and min_out is not None:
            smaller = bool(min_out < input_numel)

        rows.append(
            NodeSizeRow(
                index=idx,
                name=node.name,
                op_type=node.op_type,
                outputs=out_names,
                output_shapes=out_shapes,
                output_numels=out_numels,
                min_output_numel=min_out,
                smaller_than_input=smaller,
            )
        )

    smaller_nodes = [r for r in rows if r.smaller_than_input is True]

    summary = {
        "model": model_path,
        "shape_inference_ok": infer_ok,
        "shape_inference_error": infer_err,
        "num_nodes": len(inferred.graph.node),
        "input_tensor": input_vi.name,
        "input_shape": input_shape,
        "input_numel": input_numel,
        "smaller_nodes_count": len(smaller_nodes),
        "first_smaller_node": None if not smaller_nodes else _row_to_dict(smaller_nodes[0]),
    }

    return {
        "summary": summary,
        "smaller_nodes": [_row_to_dict(r) for r in smaller_nodes],
        "all_nodes": [_row_to_dict(r) for r in rows],
    }


def load_json(path: str) -> Tuple[Dict[str, Any], str]:
    last_err: Optional[Exception] = None
    for enc in ("utf-16", "utf-8", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f), enc
        except Exception as e:  # pragma: no cover
            last_err = e
    assert last_err is not None
    raise last_err


def fmt_node(r: Dict[str, Any]) -> str:
    name = r.get("name") or "(no-name)"
    op = r.get("op_type", "?")
    idx = r.get("index", "?")
    min_out = r.get("min_output_numel")
    shapes = r.get("output_shapes") or []
    numels = r.get("output_numels") or []
    sh0 = shapes[0] if shapes else None
    n0 = numels[0] if numels else None
    return (
        f"idx={idx:>3} op={op:<10} min_out={min_out:<8} "
        f"out0_numel={n0:<8} out0_shape={sh0} name={name}"
    )


def summarize(doc: Dict[str, Any], rep_limit: int, smallest_limit: int) -> None:
    summary = doc["summary"]
    input_numel = summary["input_numel"]

    if "all_nodes" in doc:
        all_nodes: List[Dict[str, Any]] = doc["all_nodes"]
        smaller = [
            r
            for r in all_nodes
            if r.get("min_output_numel") is not None and r["min_output_numel"] < input_numel
        ]
    else:
        smaller = list(doc.get("smaller_nodes") or [])

    smaller_sorted = sorted(smaller, key=lambda r: r["index"])
    first = smaller_sorted[0] if smaller_sorted else None

    reps: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
    for r in smaller_sorted:
        shapes = r.get("output_shapes") or []
        if not shapes:
            continue
        key = tuple(shapes[0])
        if key not in reps:
            reps[key] = r

    smallest = sorted(smaller, key=lambda r: (r["min_output_numel"], r["index"]))[:smallest_limit]

    print("[summary]")
    print(f"  model: {summary.get('model')}")
    print(f"  shape_inference_ok: {summary.get('shape_inference_ok')}")
    print(f"  num_nodes: {summary.get('num_nodes')}")
    print(f"  input_tensor: {summary.get('input_tensor')}")
    print(f"  input_shape: {summary.get('input_shape')}")
    print(f"  input_numel: {input_numel}")
    print(f"  smaller_nodes_count: {len(smaller)}")

    if first is None:
        print("\nNo nodes with output smaller than input.")
        return

    print("\n[first smaller-than-input node]")
    print("  " + fmt_node(first))

    print(f"\n[representatives by out0_shape] (limit={rep_limit})")
    for r in list(reps.values())[:rep_limit]:
        print("  " + fmt_node(r))

    print(f"\n[smallest outputs] (limit={smallest_limit})")
    for r in smallest:
        print("  " + fmt_node(r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze ONNX tensor sizes and print split-cut summary."
    )
    p.add_argument("--model", help="Path to ONNX model.")
    p.add_argument(
        "--json-path",
        help="Existing analysis JSON to summarize (skips model analysis).",
    )
    p.add_argument(
        "--save-json",
        default=None,
        help="Filename under onnx_analysis/ (default: {model_stem}_analysis.json).",
    )
    p.add_argument(
        "--no-save-json",
        action="store_true",
        help="Do not write analysis JSON when analyzing a model.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full analysis JSON to stdout instead of the text summary.",
    )
    p.add_argument("--rep-limit", type=int, default=25)
    p.add_argument("--smallest-limit", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model and not args.json_path:
        raise SystemExit("Provide --model and/or --json-path.")

    if args.json_path and not args.model:
        doc, enc = load_json(args.json_path)
        print(f"file: {args.json_path}")
        print(f"encoding: {enc}")
        print()
        if args.json:
            print(json.dumps(doc, ensure_ascii=False, indent=2))
        else:
            summarize(doc, rep_limit=args.rep_limit, smallest_limit=args.smallest_limit)
        return

    assert args.model is not None
    doc = analyze(args.model)

    if not args.no_save_json:
        DEFAULT_JSON_DIR.mkdir(parents=True, exist_ok=True)
        name = args.save_json or f"{Path(args.model).stem}_analysis.json"
        out_path = DEFAULT_JSON_DIR / Path(name).name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"wrote: {out_path}")
        print()

    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
        return

    summarize(doc, rep_limit=args.rep_limit, smallest_limit=args.smallest_limit)


if __name__ == "__main__":
    main()

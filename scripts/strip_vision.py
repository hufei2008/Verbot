#!/usr/bin/env python3
"""Strip vision tensors from Gemma 4 GGUF model, keeping only text tensors."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/llama.cpp/gguf-py"))

import gguf
from gguf.constants import GGUFValueType, GGMLQuantizationType, GGUF_QUANT_SIZES
from gguf.gguf_writer import GGUFWriter


VISION_PREFIXES = ("v.", "vision.")


def strip_vision(input_path, output_path):
    reader = gguf.GGUFReader(input_path)

    # Filter text-only tensors
    text_tensors = [t for t in reader.tensors if not t.name.startswith(VISION_PREFIXES)]

    print(
        f"Total tensors: {len(reader.tensors)} → Text: {len(text_tensors)} "
        f"(removed {len(reader.tensors) - len(text_tensors)} vision tensors)"
    )

    # Determine architecture
    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field is not None else "gemma4"

    writer = GGUFWriter(output_path, arch, endianess=reader.endianess)

    # Copy metadata fields (KV data)
    for field_name, field in reader.fields.items():
        if field_name.startswith("GGUF."):
            continue

        main_type = field.types[0] if field.types else None
        if main_type is None:
            continue

        try:
            if main_type == GGUFValueType.ARRAY:
                vals = field.contents()
                if len(vals) > 0:
                    writer.add_array(field_name, vals)
            elif main_type == GGUFValueType.STRING:
                writer.add_string(field_name, field.contents())
            elif main_type == GGUFValueType.UINT32:
                writer.add_uint32(field_name, field.contents())
            elif main_type == GGUFValueType.INT32:
                writer.add_int32(field_name, field.contents())
            elif main_type == GGUFValueType.UINT64:
                writer.add_uint64(field_name, field.contents())
            elif main_type == GGUFValueType.INT64:
                writer.add_int64(field_name, field.contents())
            elif main_type == GGUFValueType.FLOAT32:
                writer.add_float32(field_name, field.contents())
            elif main_type == GGUFValueType.FLOAT64:
                writer.add_float64(field_name, field.contents())
            elif main_type == GGUFValueType.BOOL:
                writer.add_bool(field_name, field.contents())
            else:
                print(f"  Skip field {field_name} with type {main_type}")
        except Exception as e:
            print(f"  Error copying field {field_name}: {e}")

    # Add text-only tensors
    for tensor in text_tensors:
        data = tensor.data.copy()
        if data.dtype == np.float16:
            data = data.astype(np.float32)

        gtype = tensor.tensor_type

        # For non-quantized types, simplest path
        if gtype in (
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
            GGMLQuantizationType.F64,
            GGMLQuantizationType.I8,
            GGMLQuantizationType.I16,
            GGMLQuantizationType.I32,
            GGMLQuantizationType.I64,
        ):
            writer.add_tensor(tensor.name, data)
            continue

        # For quantized types, pass raw_dtype + raw_shape = logical shape from reader (not byte shape)
        # tensor.shape is the logical shape (from tensor info), NOT reversed
        # For quantized types, data is uint8 with byte shape. The writer will convert back to logical.
        # We pass the logical shape (tensor.shape) so that add_tensor_info can do the shape math correctly.
        writer.add_tensor(
            tensor.name,
            data,
            raw_shape=tuple(reversed(tensor.shape.tolist())),  # logical shape (reversed to original order)
            raw_dtype=gtype,
        )

    # Write header, KV data, tensor info, and tensor data
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()

    print(f"Done! Written to: {output_path}")


if __name__ == "__main__":
    input_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "models/gemma4-26b-a4b-it-q4_k_m.gguf"
    )
    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else input_path.replace(".gguf", "-text.gguf")
    )

    if "--list" in sys.argv:
        reader = gguf.GGUFReader(input_path)
        vision_count = sum(
            1 for t in reader.tensors if t.name.startswith(VISION_PREFIXES)
        )
        text_count = len(reader.tensors) - vision_count
        print(f"Total tensors: {len(reader.tensors)}")
        print(f"Text tensors: {text_count}")
        print(f"Vision tensors: {vision_count}")
    else:
        strip_vision(input_path, output_path)
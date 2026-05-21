#!/usr/bin/env python3
"""Strip vision tensors from Gemma 4 GGUF model, keeping only text tensors."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/llama.cpp/gguf-py"))

import gguf

VISION_PREFIXES = ("v.", "vision.")

def strip_vision(input_path, output_path):
    reader = gguf.GGUFReader(input_path)
    
    # Filter text-only tensors
    text_tensors = [t for t in reader.tensors if not t.name.startswith(VISION_PREFIXES)]
    orig_meta = reader.fields
    
    print(f"Total tensors: {len(reader.tensors)} → Text: {len(text_tensors)} (removed {len(reader.tensors) - len(text_tensors)} vision tensors)")
    
    # Create writer using the same architecture
    # First determine the architecture from the metadata
    arch = None
    for name in ("general.architecture",):
        if name in reader.fields:
            arch = str(reader.fields[name].value)
            break
    
    if arch:
        writer = gguf.GGUFWriter(output_path, arch)
    else:
        writer = gguf.GGUFWriter(output_path, "gemma4")
    
    # Copy all metadata key-value pairs from original, preserving their values
    # We need to iterate through the original GGUF fields and add them to the new file
    for field_name, field in reader.fields.items():
        # Skip fields whose names look like tensor names
        # tensor-related metadata keys will be handled by add_tensor
        if any(field_name.startswith(p) for p in ("blk.", "token_embd", "output_norm", "output.", "rope.", "v.", "vision.")):
            continue
        
        values = field.values
        if len(values) == 1:
            v = values[0]
            if isinstance(v, (int, np.integer)):
                writer.add_key(field_name, int(v))
            elif isinstance(v, (float, np.floating)):
                writer.add_key(field_name, float(v))
            elif isinstance(v, (str, bytes)):
                s = str(v) if isinstance(v, str) else v.decode("utf-8", errors="replace")
                writer.add_key(field_name, s)
            elif isinstance(v, np.ndarray) and v.ndim == 1:
                # Convert numpy array to list for add_array
                if v.dtype.kind == 'i':
                    writer.add_array(field_name, [int(x) for x in v])
                elif v.dtype.kind == 'f':
                    writer.add_array(field_name, [float(x) for x in v])
                elif v.dtype.kind in ('U', 'S', 'O'):
                    writer.add_array(field_name, [str(x) for x in v])
                elif v.dtype.kind == 'b':
                    writer.add_array(field_name, [bool(x) for x in v])
                else:
                    print(f"  Warning: skipping field {field_name} with dtype {v.dtype}")
            else:
                print(f"  Warning: skipping field {field_name} with type {type(v)}")
        else:
            # Multi-value field (actual array in GGUF terms)
            if all(isinstance(v, (int, np.integer)) for v in values):
                writer.add_array(field_name, [int(v) for v in values])
            elif all(isinstance(v, (float, np.floating)) for v in values):
                writer.add_array(field_name, [float(v) for v in values])
            elif all(isinstance(v, (str, bytes)) for v in values):
                writer.add_array(field_name, [str(v) if isinstance(v, str) else v.decode("utf-8", errors="replace") for v in values])
            else:
                types = set(type(v).__name__ for v in values)
                print(f"  Warning: skipping multi-value field {field_name} with types {types}")
    
    # Write header and KV data
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    
    # Write tensor data for text-only tensors
    for tensor in text_tensors:
        data = tensor.data
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        
        writer.add_tensor(tensor.name, data)
    
    # Close and finalize
    writer.close()
    
    print(f"Done! Written to: {output_path}")

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "models/gemma4-26b-a4b-it-q4_k_m.gguf"
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".gguf", "-text.gguf")
    
    if "--list" in sys.argv:
        reader = gguf.GGUFReader(input_path)
        vision_count = sum(1 for t in reader.tensors if t.name.startswith(VISION_PREFIXES))
        text_count = len(reader.tensors) - vision_count
        print(f"Total tensors: {len(reader.tensors)}")
        print(f"Text tensors: {text_count}")
        print(f"Vision tensors: {vision_count}")
    else:
        strip_vision(input_path, output_path)
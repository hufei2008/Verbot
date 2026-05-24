#!/usr/bin/env python3
# ============================================================
# Gemma 4 GGUF 模型视觉张量剥离工具
#
# 功能：从一个包含视觉（vision）和文本（text）张量的 GGUF 模型中，
#       移除所有视觉相关张量，仅保留文本张量，输出一个新的 GGUF 文件。
#
# 用途：Gemma 4 是多模态模型，视觉张量很大但某些纯文本场景不需要，
#       此工具可以显著减小模型体积。
#
# 用法：
#   python3 scripts/strip_vision.py <input.gguf> [output.gguf]
#   python3 scripts/strip_vision.py model.gguf          # 输出 model-text.gguf
#   python3 scripts/strip_vision.py model.gguf --list   # 仅列统计
#
# 依赖：需要 llama.cpp 的 gguf-py 模块（位于 vendor/llama.cpp/gguf-py/）
# ============================================================

import sys
import numpy as np
from pathlib import Path

# 将 llama.cpp 的 gguf-py 模块加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor/llama.cpp/gguf-py"))

import gguf
from gguf.constants import GGUFValueType, GGMLQuantizationType, GGUF_QUANT_SIZES
from gguf.gguf_writer import GGUFWriter


# 视觉张量的名称前缀，用于识别和过滤
VISION_PREFIXES = ("v.", "vision.")


def strip_vision(input_path, output_path):
    # type: (str, str) -> None
    """读取 GGUF 模型，移除视觉张量，写入新的纯文本 GGUF 文件。

    Args:
        input_path:  输入 GGUF 文件路径
        output_path: 输出 GGUF 文件路径
    """
    # 打开并解析输入 GGUF 文件
    reader = gguf.GGUFReader(input_path)

    # 过滤出非视觉张量（名称不以 v. 或 vision. 开头）
    text_tensors = [t for t in reader.tensors if not t.name.startswith(VISION_PREFIXES)]

    print(
        f"Total tensors: {len(reader.tensors)} → Text: {len(text_tensors)} "
        f"(removed {len(reader.tensors) - len(text_tensors)} vision tensors)"
    )

    # 从元数据中获取模型架构信息
    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field is not None else "gemma4"

    # 创建 GGUF 写入器，保持与输入相同的字节序
    writer = GGUFWriter(output_path, arch, endianess=reader.endianess)

    # ---- 复制元数据字段（KV 数据） ----
    # 遍历所有元数据字段，按类型复制到输出文件
    for field_name, field in reader.fields.items():
        # 跳过 GGUF 内部字段（以 "GGUF." 开头）
        if field_name.startswith("GGUF."):
            continue

        # 获取字段的主类型
        main_type = field.types[0] if field.types else None
        if main_type is None:
            continue

        try:
            # 根据字段类型调用对应的写入方法
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

    # ---- 添加纯文本张量 ----
    for tensor in text_tensors:
        data = tensor.data.copy()

        # float16 转 float32：GGUF 规范中 f16 需转为 f32 再写入
        if data.dtype == np.float16:
            data = data.astype(np.float32)

        gtype = tensor.tensor_type

        # 非量化类型：直接添加张量，无需额外处理
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

        # 量化类型：需要传递 raw_dtype 和 raw_shape
        # tensor.shape 是逻辑形状（来自 tensor info），不是字节形状
        # 量化数据以 uint8 存储，字节形状需要反转以恢复原始顺序
        writer.add_tensor(
            tensor.name,
            data,
            raw_shape=tuple(reversed(tensor.shape.tolist())),  # 逻辑形状（反转回原始顺序）
            raw_dtype=gtype,
        )

    # ---- 写入输出文件 ----
    # 步骤1：写入文件头（GGUF 魔数、版本等）
    writer.write_header_to_file()
    # 步骤2：写入键值元数据
    writer.write_kv_data_to_file()
    # 步骤3：写入张量信息（名称、形状、类型、偏移量）
    writer.write_ti_data_to_file()
    # 步骤4：关闭文件（写入张量数据并刷新）
    writer.close()

    print(f"Done! Written to: {output_path}")


if __name__ == "__main__":
    # 解析命令行参数
    # 第1个参数：输入文件路径（默认：models/gemma4-26b-a4b-it-q4_k_m.gguf）
    input_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "models/gemma4-26b-a4b-it-q4_k_m.gguf"
    )
    # 第2个参数：输出文件路径（默认：输入文件名 + "-text.gguf"）
    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else input_path.replace(".gguf", "-text.gguf")
    )

    # --list 模式：仅统计张量数量，不执行剥离
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
        # 执行视觉张量剥离
        strip_vision(input_path, output_path)
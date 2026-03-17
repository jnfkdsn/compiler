"""Shared kernel ABI status codes and helpers."""

from __future__ import annotations

from enum import IntEnum


class AbiStatus(IntEnum):
    OK = 0

    INPUT_COUNT_MISMATCH = 100
    OUT_DESC_NULL = 101
    OUT_DATA_NULL = 102
    OUT_NUMEL_MISMATCH = 103
    OUT_RANK_MISMATCH = 104
    OUT_SHAPE_MISMATCH = 105
    OUT_STRIDE_MISMATCH = 106

    INPUT_DATA_NULL_BASE = 200
    INPUT_NUMEL_MISMATCH_BASE = 300
    INPUT_RANK_MISMATCH_BASE = 400
    INPUT_SHAPE_MISMATCH_BASE = 500
    INPUT_STRIDE_MISMATCH_BASE = 600


def decode_abi_status(status: int) -> str:
    if status == AbiStatus.OK:
        return "ok"
    if status == AbiStatus.INPUT_COUNT_MISMATCH:
        return "input_count_mismatch"
    if status == AbiStatus.OUT_DESC_NULL:
        return "out_desc_null"
    if status == AbiStatus.OUT_DATA_NULL:
        return "out_data_null"
    if status == AbiStatus.OUT_NUMEL_MISMATCH:
        return "out_numel_mismatch"
    if status == AbiStatus.OUT_RANK_MISMATCH:
        return "out_rank_mismatch"
    if status == AbiStatus.OUT_SHAPE_MISMATCH:
        return "out_shape_mismatch"
    if status == AbiStatus.OUT_STRIDE_MISMATCH:
        return "out_stride_mismatch"

    if status >= AbiStatus.INPUT_DATA_NULL_BASE and status < AbiStatus.INPUT_NUMEL_MISMATCH_BASE:
        return "input_data_null"
    if status >= AbiStatus.INPUT_NUMEL_MISMATCH_BASE and status < AbiStatus.INPUT_RANK_MISMATCH_BASE:
        return "input_numel_mismatch"
    if status >= AbiStatus.INPUT_RANK_MISMATCH_BASE and status < AbiStatus.INPUT_SHAPE_MISMATCH_BASE:
        return "input_rank_mismatch"
    if status >= AbiStatus.INPUT_SHAPE_MISMATCH_BASE and status < AbiStatus.INPUT_STRIDE_MISMATCH_BASE:
        return "input_shape_mismatch"
    if status >= AbiStatus.INPUT_STRIDE_MISMATCH_BASE and status < AbiStatus.INPUT_STRIDE_MISMATCH_BASE + 100:
        return "input_stride_mismatch"
    return "unknown_abi_status"

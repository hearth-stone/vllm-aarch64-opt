# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compressed_slot_mapping_kernel(
    # [num_tokens]
    slot_mapping_ptr,
    # [num_reqs + 1]
    query_start_loc_ptr,
    # [num_reqs]
    seq_lens_ptr,
    # [num_reqs, max_num_blocks]
    block_table_ptr,
    block_table_stride,
    block_size,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        is_valid = (pos + 1) % COMPRESS_RATIO == 0
        pos_after_compress = pos // COMPRESS_RATIO

        block_ids = pos_after_compress // block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid,
        )
        slot_ids = block_numbers * block_size + pos_after_compress % block_size

        # NOTE
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    if query_start_loc.device.type == "cpu":
        # Pure-torch port of ``_compressed_slot_mapping_kernel``. The kernel
        # writes ``slot_mapping[query_start_loc[r] + i]`` for every query
        # token ``i`` in request ``r``, computing a compressed-cache slot
        # id where applicable and -1 (PAD_ID) otherwise. Vectorize across
        # tokens via boolean masks instead of a Python per-request loop.
        if num_reqs > 0 and num_tokens > 0:
            qsl_cpu = query_start_loc.to(torch.long)
            sl_cpu = seq_lens.to(torch.long)
            # ``token_to_req[t]`` = which request token ``t`` belongs to.
            query_lens = qsl_cpu[1:] - qsl_cpu[:-1]
            req_ids = torch.repeat_interleave(
                torch.arange(num_reqs, device=query_start_loc.device), query_lens
            )
            actual = req_ids.shape[0]
            # ``start_pos[r] = seq_len[r] - query_len[r]`` is the absolute
            # position of the first query token in request ``r``.
            start_pos = sl_cpu - query_lens
            # Per-token absolute position within its sequence.
            token_idx_in_req = (
                torch.arange(actual, device=query_start_loc.device)
                - qsl_cpu[req_ids]
            )
            pos = start_pos[req_ids] + token_idx_in_req

            is_valid = ((pos + 1) % compress_ratio) == 0
            pos_after_compress = pos // compress_ratio
            block_ids = pos_after_compress // block_size
            block_offset = pos_after_compress % block_size
            # ``block_table[req, block_id]`` per token; mask invalid entries
            # to block 0 to avoid OOB before the final ``where``.
            safe_block_ids = torch.where(
                is_valid, block_ids, torch.zeros_like(block_ids)
            )
            block_numbers = block_table[req_ids, safe_block_ids].to(torch.long)
            slot_ids = block_numbers * block_size + block_offset
            slot_ids = torch.where(
                is_valid, slot_ids, torch.full_like(slot_ids, -1)
            )
            slot_mapping[:actual] = slot_ids
        return slot_mapping

    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        compress_ratio,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping

# Applying Flash-Decoding as descibed in
# https://pytorch.org/blog/flash-decoding/
# by Tri Dao, 2023

# based on the code of Yuanheng Zhao (Colossal-Infer)
import torch
import triton
import triton.language as tl


# Triton 2.1.0
@triton.jit
def _flash_decoding_fwd_kernel(
    Q,  # [batch_size, head_num, q_len, head_dim]
    KCache,  # [num_blocks, num_kv_heads, block_size, head_dim]
    VCache,  # [num_blocks, num_kv_heads, block_size, head_dim]
    mid_o,  # [batch_size * q_len, head_num, kv_split_num, head_dim]
    mid_o_lse,  # [batch_size * q_len, head_num, kv_split_num]
    kv_seq_len,  # [batch_size]
    q_len: tl.constexpr,
    batch_size,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_q_qlen,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_mid_ot,
    stride_mid_oh,
    stride_mid_ob,
    stride_mid_oqlen,
    stride_mid_od,
    stride_mid_o_lset,
    stride_mid_o_lseh,
    stride_mid_o_lseb,
    KV_GROUPS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    cur_token_idx = tl.program_id(0)

    cur_head_idx = tl.program_id(1) # head
    block_start_kv = tl.program_id(2)  # kv block, for splitting k/v
    # get the current (kv) sequence length
    # cur_token_off is used as a "mask" here for spec-dec during verification process
    cur_kv_seq_len = tl.load(kv_seq_len + cur_token_idx)
    if block_start_kv * BLOCK_KV >= cur_kv_seq_len:
        return

    offsets_dmodel = tl.arange(0, HEAD_DIM)
    offsets_q = cur_token_idx * stride_qt + cur_head_idx * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + offsets_q,
        shape=(q_len, HEAD_DIM),
        strides=(stride_q_qlen, stride_qd),
        offsets=(0, 0),
        block_shape=(q_len, HEAD_DIM),
        order=(0, 1),
    )
    q = tl.load(Q_block_ptr)
    # head dim
    cur_kv_head_idx = cur_head_idx // KV_GROUPS
    cur_k_offset = cur_token_idx * stride_kb + cur_kv_head_idx * stride_kh + block_start_kv * BLOCK_KV * stride_kt
    cur_v_offset = cur_token_idx * stride_vb + cur_kv_head_idx * stride_vh + block_start_kv * BLOCK_KV * stride_vt

    K_block_ptr = tl.make_block_ptr(
        base=KCache + cur_k_offset,
        shape=(cur_kv_seq_len, HEAD_DIM),
        strides=(stride_kd, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_KV),
        order=(1, 0),
    )
    # block, headdim
    V_block_ptr = tl.make_block_ptr(
        base=VCache + cur_v_offset,
        shape=(cur_kv_seq_len, HEAD_DIM),
        strides=(stride_vt, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(0, 1),
    )
    block_mask = block_start_kv * BLOCK_KV +  tl.arange(0, BLOCK_KV) < cur_kv_seq_len
    k_cur_block = tl.load(K_block_ptr)
    v_cur_block = tl.load(V_block_ptr)

    acc = tl.zeros([q_len, HEAD_DIM], dtype=tl.float32)
    # use block size of the paged/blocked kv cache
    S_ij = tl.zeros([q_len, BLOCK_KV], dtype=tl.float32)

    # print(f"grid: ({(tl.program_id(0), tl.program_id(1), tl.program_id(2))})")
    # print(f"q is {q}, sum is {tl.sum(q)}")
    # print(f"k is {k_cur_block}, sum is {tl.sum(k_cur_block)}")
    # print(f"v is {v_cur_block}, sum is {tl.sum(v_cur_block)}")
    # print('--------')

    # NOTE a trick to come across triton's requirement that values in both first and second input shapes must be >= 16,
    # Multiplying two tensors with shapes [1, d] * [d, block_size] will fail.
    # Refer to https://github.com/openai/triton/discussions/895
    # S_ij += tl.sum(q[None, :] * k_cur_block, 1) # 1 * d, blocksize * d -> block*256 -> block * 1
    S_ij += tl.dot(q, k_cur_block)
    S_ij = tl.where(block_mask[None, :], S_ij, float("-inf"))
    S_ij *= sm_scale
    m = tl.max(S_ij, 1)
    S_ij -= m[:, None]
    p_ij_hat = tl.exp(S_ij)
    l_i = tl.sum(p_ij_hat, 1)
    p_ij_hat = p_ij_hat.to(v_cur_block.type.element_ty)# q_len, block_kv
    # v_cur_block: block_kv, head_dim
    # p_ij_hat: q_len, block_kv
    acc += tl.dot(p_ij_hat, v_cur_block)
    acc = acc / l_i[:, None]

    # (bsz, num_heads, kv_max_split_num, qlen, head_dim
    cur_offest_mid = cur_token_idx * stride_mid_ot \
                    + cur_head_idx * stride_mid_oh \
                    + block_start_kv * stride_mid_ob
    
    offsets_mid_o = tl.make_block_ptr(
        base=mid_o + cur_offest_mid,
        shape=(q_len, HEAD_DIM),
        strides=(stride_mid_oqlen, stride_mid_od),
        offsets=(0, 0),
        block_shape=(q_len, HEAD_DIM),
        order=(0, 1),
    )
    tl.store(offsets_mid_o, acc)

    # BUG
    offsets_qlen = tl.arange(0, q_len)
    offsets_mid_o_lse = (
        cur_token_idx * stride_mid_o_lset + cur_head_idx * stride_mid_o_lseh + block_start_kv * stride_mid_o_lseb + offsets_qlen
    )
    # logsumexp l_i^(j) = m^(j) + log(l_i^(j))
    tl.store(mid_o_lse + offsets_mid_o_lse, m + tl.log(l_i))

# Triton 2.1.0
@triton.jit
def _flash_decoding_fwd_reduce_kernel(
    mid_o,  # [batch_size, head_num, q_len kv_split_num, head_dim]
    mid_o_lse,  # [batch_size, head_num, q_len, kv_split_num]
    O,  # [batch_size, num_heads, head_dim] or [batch_size, 1, num_heads, head_dim]
    kv_seq_len,
    q_len: tl.constexpr,
    batch_size,
    stride_mid_ot,
    stride_mid_oh,
    stride_mid_ob,
    stride_mid_oqlen,
    stride_mid_od,
    stride_o_lset,
    stride_o_lseh,
    stride_o_lseb,
    stride_o_lseqlen,
    stride_ot,
    stride_oh,
    stride_oqlen,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    cur_token_idx = tl.program_id(0)

    cur_head_idx = tl.program_id(1)
    cur_q_idx = tl.program_id(2)

    # cur_token_off is used as a "mask" here for spec-dec during verification process
    cur_kv_seq_len = tl.load(kv_seq_len + cur_token_idx)
    offsets_dmodel = tl.arange(0, HEAD_DIM)

    # NOTE currently the block size BLOCK_KV splitting kv is relatively small as we have
    # BLOCK_KV == BLOCK_SIZE for now. We might want to decrease the number of blocks of kv splitted.
    kv_split_num = (cur_kv_seq_len + BLOCK_KV - 1) // BLOCK_KV
    m_i = float("-inf")  # max logic
    l_i = 0.0  # sum exp
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # (bsz, num_heads, kv_max_split_num, qlen, head_dim
    offsets_mid_o = cur_token_idx * stride_mid_ot + cur_head_idx * stride_mid_oh + cur_q_idx * stride_mid_oqlen + offsets_dmodel
    offset_mid_lse = cur_token_idx * stride_o_lset + cur_head_idx * stride_o_lseh + cur_q_idx * stride_o_lseqlen
    for block_i in range(0, kv_split_num, 1):
        mid_o_block = tl.load(mid_o + offsets_mid_o + block_i * stride_mid_ob)
        lse = tl.load(mid_o_lse + offset_mid_lse + block_i * stride_o_lseb)
        m_ij = tl.maximum(m_i, lse)
        scale = tl.exp(m_i - m_ij)
        acc = acc * scale
        lse -= m_ij
        exp_logic = tl.exp(lse)
        acc += exp_logic * mid_o_block
        l_i = scale * l_i + exp_logic
        m_i = m_ij

    acc = acc / l_i
    offsets_O = cur_token_idx * stride_ot + cur_head_idx * stride_oh + cur_q_idx * stride_oqlen + offsets_dmodel
    tl.store(O + offsets_O, acc.to(O.type.element_ty))
    return l_i


# Decoding Stage
def flash_decoding_attention(
    q: torch.Tensor, # bsz H qlen
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_seq_len: torch.Tensor,
    block_size: int = 64,
    max_seq_len_in_batch: int = None,
    output: torch.Tensor = None,
    mid_output: torch.Tensor = None,
    mid_output_lse: torch.Tensor = None,
    sm_scale: int = None,
    kv_group_num: int = 1,
):
    """
    Flash decoding implemented with a blocked KV Cache (PagedAttention) during decoding stage.

    Args:
        q (torch.Tensor): [bsz, num_heads, q_len, head_dim]
            q_len > 1 only for verification process in speculative-decoding.
        k_cache (torch.Tensor): [bsz, num_kv_heads, ctx, head_dim]
        v_cache (torch.Tensor): [bsz, num_kv_heads, ctx, head_dim]
        kv_seq_len (torch.Tensor): [batch_size]
            records the (kv) sequence lengths incorporating past kv sequence lengths.
        max_seq_len_in_batch (int): Maximum sequence length in the batch.
        output (torch.Tensor):  [bsz, num_heads * head_dim]
        mid_output (torch.Tensor): [max_bsz, num_heads, q_len, kv_max_split_num, head_dim]
            Intermediate output tensor. `max_bsz` should be greater than or equal to `bsz`.
        mid_output_lse (torch.Tensor): [max_bsz, num_heads, q_len, kv_max_split_num]
            Log-sum-exp of intermediate output. `max_bsz` should be greater than or equal to `bsz`.
        block_size (int): Size of each splitted kv.
        num_kv_group (int, optional): Number of key/value groups. Defaults to 1.

    Returns:
        Output tensor with shape [bsz * q_len, num_heads * head_dim]
    """
    # q = q.squeeze() if q.dim() == 4 else q
    # assert q.dim() == 3, f"Incompatible q dim: {q.dim()}"
    n_tokens, num_heads, q_len, head_dim = q.shape
    q_len = int(q_len)
    bsz = n_tokens

    assert head_dim in {32, 64, 128, 256}
    assert k_cache.size(-2) == v_cache.size(-2), f"Got incompatible block size on kv caches:\n"

    # NOTE BLOCK_KV could be considered as block splitting the sequence on k/v
    # For now, BLOCK_KV is supposed to be equivalent with the size of physical cache block (i.e.`block_size`)
    assert block_size in {16, 32, 64, 128}
    BLOCK_KV = block_size

    sm_scale = 1.0 / (head_dim**0.5) if sm_scale is None else sm_scale
    max_seq_len_in_batch = kv_seq_len.max().item() if max_seq_len_in_batch is None else max_seq_len_in_batch
    # For compatibility (TODO revise modeling in future)
    kv_max_split_num = (max_seq_len_in_batch + BLOCK_KV - 1) // BLOCK_KV

    if mid_output is None:
        mid_output = torch.empty(
            (bsz, num_heads, kv_max_split_num, q_len, head_dim), dtype=torch.float32, device=q.device
        )
    if mid_output_lse is None:
        mid_output_lse = torch.empty((bsz, num_heads, kv_max_split_num, q_len), dtype=torch.float32, device=q.device)
    if output is None:
        # A hack to prevent `view` operation in modeling
        output = torch.empty((bsz, num_heads, q_len, head_dim), dtype=q.dtype, device=q.device)

    assert (
        mid_output.size(2) == mid_output_lse.size(2) >= kv_max_split_num
    ), "Incompatible kv split number of intermediate output tensors"
    assert (
        mid_output.size(0) == mid_output_lse.size(0) >= output.size(0) == n_tokens
    ), f"Incompatible first dimension of output tensors"

    # NOTE use `triton.next_power_of_2` here to utilize the cache mechanism of triton
    # To optimize, revise batching/scheduling to batch 2^n sequences in a batch (preferred)
    grid = lambda META: (
        triton.next_power_of_2(bsz),
        num_heads,
        triton.cdiv(triton.next_power_of_2(max_seq_len_in_batch), META["BLOCK_KV"]),
    )




    _flash_decoding_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        mid_output,
        mid_output_lse,
        kv_seq_len,
        q_len,
        bsz,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        mid_output.stride(3),
        mid_output.stride(4),
        mid_output_lse.stride(0),
        mid_output_lse.stride(1),
        mid_output_lse.stride(2),
        KV_GROUPS=kv_group_num,
        BLOCK_KV=block_size,
        HEAD_DIM=head_dim,
    )
    # mid_output = torch.permute(mid_output, (0, 1, 3, 2, 4)).contiguous()
    # assert torch.allclose(mid_output[:, :, 0] , mid_output[:, :, 1], atol=1e-2, rtol=0)
    # assert torch.allclose(mid_output_lse[:, :, 0] , mid_output_lse[:, :, 1], atol=1e-2, rtol=0)
    # mid_output_lse = torch.permute(mid_output_lse, (0, 1, 3, 2)).contiguous()
    grid = (triton.next_power_of_2(bsz), num_heads, q_len)
    _flash_decoding_fwd_reduce_kernel[grid](
        mid_output,
        mid_output_lse,
        output,
        kv_seq_len,
        q_len,
        bsz,
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        mid_output.stride(3),
        mid_output.stride(4),
        mid_output_lse.stride(0),
        mid_output_lse.stride(1),
        mid_output_lse.stride(2),
        mid_output_lse.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_KV=block_size,
        HEAD_DIM=head_dim,
    )

    return output

import pytest

def torch_attention(q, k, v, sm_scale):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()
    qk_ref_out = torch.matmul(p, v)
    return qk_ref_out


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, q_len", [(32, 4, 4096, 128, 128)])
def test_op(Z, H, N_CTX, HEAD_DIM, q_len):
    dtype=torch.float16
    # torch.manual_seed(20)
    q = (torch.empty((Z, H, q_len, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))

    sm_scale = 1.0 / (HEAD_DIM**0.5)
    # reference implementation
    qk_ref_out = torch_attention(q, k, v, sm_scale)
    kv_seq_len = torch.tensor([N_CTX] * Z).cuda()
    # force right padding
    tri_out = flash_decoding_attention(q, k, v, kv_seq_len, 128).half()
    assert  torch.allclose(qk_ref_out, tri_out.view(Z, H, q_len, HEAD_DIM), atol=1e-2, rtol=0)

BATCH, N_HEADS, HEAD_DIM, Q_LEN = 4, 32, 128, 128
# vary seq length for fixed head and batch=4
configs = []
# TODO: padding
# TODO: KV Cache
# TODO:
for Q_LEN in [16, 32, 64, 128, 256]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(10, 18)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["triton", "torch"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"flash-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-qlen{Q_LEN}",
            args={
                "H": N_HEADS,
                "Z": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "q_len": Q_LEN,
            },
        ))
    
@triton.testing.perf_report(configs)
def benchmark(Z, H, N_CTX, HEAD_DIM, q_len, provider):
    dtype=torch.float16
    # torch.manual_seed(20)
    q = (torch.empty((Z, H, q_len, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        sm_scale = 1.0 / (HEAD_DIM**0.5)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_attention(q, k, v, sm_scale), quantiles=quantiles)
    if provider == 'triton':
        kv_seq_len = torch.tensor([N_CTX] * Z).cuda()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_decoding_attention(q, k, v, kv_seq_len, 128), quantiles=quantiles)
    gbps = lambda ms: (q.numel() * q.element_size() + k.numel() * k.element_size() + v.numel() * v.element_size()) / ms * 1e-6
    return ms, min_ms, max_ms #gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=False)

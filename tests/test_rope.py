import pytest
import torch

from tite.model.pool import PackedMetaData
from tite.model.rope import RotaryPositionalEmbeddings


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
def test_rope(dtype: torch.dtype):
    batch_size = 2
    seq_len = 128
    num_attention_heads = 12
    head_dim = 64
    hidden_states = torch.rand(batch_size * seq_len, num_attention_heads, head_dim, device="cuda", dtype=dtype)
    eager_rope = RotaryPositionalEmbeddings(head_dim, implementation="eager").to("cuda")
    rope = RotaryPositionalEmbeddings(head_dim).to("cuda")

    seq_lens = torch.tensor([seq_len] * batch_size, device="cuda", dtype=torch.int32)
    cu_seq_lens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
    cu_seq_lens[1:] = seq_lens.cumsum(0)
    packed_meta_data = PackedMetaData(seq_lens, cu_seq_lens, seq_len, None)

    eager_rope_hidden_states = eager_rope(hidden_states, packed_meta_data)
    rope_hidden_states = rope(hidden_states, packed_meta_data)

    assert torch.allclose(eager_rope_hidden_states, rope_hidden_states, atol=1e-6)


def test_rope_pooling():
    num_attention_heads = 12
    head_dim = 64
    seq_len = 256
    rope = RotaryPositionalEmbeddings(head_dim, implementation="eager").to("cuda")
    hidden_states = torch.rand(seq_len, num_attention_heads, head_dim, device="cuda")

    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    cu_seq_lens = torch.zeros(2, device="cuda", dtype=torch.int32)
    cu_seq_lens[1] = seq_len
    packed_meta_data = PackedMetaData(seq_lens, cu_seq_lens, seq_len, None)

    rope_hidden_states = rope(hidden_states, packed_meta_data, kernel_size=2, stride=2)

    # assert torch.allclose(rope_hidden_states[0], rope_hidden_states[1][:128], atol=1e-6)


def test_rope_offset():
    num_attention_heads = 12
    head_dim = 64
    rope = RotaryPositionalEmbeddings(head_dim, implementation="eager").to("cuda")
    hidden_states = torch.rand(256, num_attention_heads, head_dim, device="cuda")

    rope_hidden_states = []
    for seq_len in [128, 256]:
        seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
        cu_seq_lens = torch.zeros(2, device="cuda", dtype=torch.int32)
        cu_seq_lens[1] = seq_len
        packed_meta_data = PackedMetaData(seq_lens, cu_seq_lens, seq_len, None)

        rope_hidden_states.append(rope(hidden_states[:seq_len], packed_meta_data))

    assert torch.allclose(rope_hidden_states[0], rope_hidden_states[1][:128], atol=1e-6)

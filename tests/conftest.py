from pathlib import Path

import pytest

import tite.transformation.string_transformation as string_transformation
import tite.transformation.token_transformation as token_transformation
from tite.model.tite import TiteConfig
from tite.model.tokenizer import TiteTokenizer

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def tokenizer() -> TiteTokenizer:
    tokenizer = TiteTokenizer.from_pretrained(DATA_DIR / "tokenizer")
    return tokenizer


@pytest.fixture
def config() -> TiteConfig:
    config = TiteConfig(
        vocab_size=32,
        num_hidden_layers=3,
        hidden_sizes=(4, 8, 12),
        num_attention_heads=(2, 2, 2),
        intermediate_sizes=(8, 12, 16),
        kernel_sizes=(8, 8, None),
        strides=(2, 1, None),
        max_position_embeddings=16,
        rotary_interleaved=True,
        positional_embedding_type="rotary",
        attn_implementation="eager",
        rope_implementation="eager",
        pooling_implementation="eager",
        compile=False,
    )
    return config


@pytest.fixture
def string_transformations():
    return {
        "contrastive_learning": [string_transformation.CharacterSwapNeighboring()],
        "windowed": [string_transformation.CharacterInsert()],
    }


@pytest.fixture
def token_transformations():
    return {
        "contrastive_learning": [token_transformation.TokenMask(mask_id=103, mask_prob=0.3)],
        "windowed": [token_transformation.TokenInsert(vocab_size=32, pad_id=0)],
    }

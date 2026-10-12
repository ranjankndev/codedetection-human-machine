import pytest

torch = pytest.importorskip("torch")


class FakeTokenizer:
    """Char-level tokenizer with the 3 methods the code uses; no downloads needed."""
    pad_token_id, cls_id, sep_id = 1, 0, 2

    def encode(self, text, add_special_tokens=True, **kw):
        ids = [(ord(c) % 90) + 5 for c in text]
        return self.build_inputs_with_special_tokens(ids) if add_special_tokens else ids

    def build_inputs_with_special_tokens(self, ids):
        return [self.cls_id] + list(ids) + [self.sep_id]


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.fixture
def tiny_encoder():
    from transformers import RobertaConfig, RobertaModel
    torch.manual_seed(0)
    cfg = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=64, max_position_embeddings=140, pad_token_id=1, type_vocab_size=1)
    return RobertaModel(cfg)

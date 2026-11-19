import torch

from moxley.config import MoxleyConfig
from moxley.model import MoxleyForCausalLM


def tiny_config() -> MoxleyConfig:
    return MoxleyConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=512,
        max_position_embeddings=128,
    )


def test_forward_and_loss():
    torch.manual_seed(0)
    model = MoxleyForCausalLM(tiny_config())
    x = torch.randint(0, 512, (2, 12))
    output = model(x, labels=x)
    assert output.logits.shape == (2, 12, 512)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.loss.item() > 0
    assert output.past_key_values is None

    # Tied embeddings must share storage, which is what keeps the model at 64M.
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_kv_cache_matches_full_forward():
    torch.manual_seed(1)
    model = MoxleyForCausalLM(tiny_config()).eval()
    x = torch.randint(0, 512, (2, 12))
    with torch.no_grad():
        full_logits = model(x).logits
        prefix = model(x[:, :8], use_cache=True)
        step = model(
            x[:, 8:9],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
    assert torch.allclose(step.logits[:, -1], full_logits[:, 8], atol=1e-5)


def test_generate():
    torch.manual_seed(2)
    model = MoxleyForCausalLM(tiny_config()).eval()
    prompt = torch.randint(0, 512, (1, 4))
    output = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
    )
    assert output.shape == (1, 9)
    assert torch.equal(output[:, :4], prompt)


def test_save_and_load(tmp_path):
    torch.manual_seed(3)
    model = MoxleyForCausalLM(tiny_config()).eval()
    model.save_pretrained(tmp_path)
    loaded = MoxleyForCausalLM.from_pretrained(tmp_path).eval()
    x = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        assert torch.allclose(model(x).logits, loaded(x).logits)

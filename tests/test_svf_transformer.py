import torch

from models import SVFTransformer, SVFTransformerConfig


def test_forward_backward_smoke():
    config = SVFTransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        memory_size=8,
    )
    model = SVFTransformer(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))

    out = model(x, targets=y, write_memory=True)

    assert out.logits.shape == (2, 16, config.vocab_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss)

    out.loss.backward()
    grad_norm = sum(
        p.grad.detach().abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    assert grad_norm > 0


def test_generate_shape():
    config = SVFTransformerConfig(
        vocab_size=16,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        max_seq_len=8,
    )
    model = SVFTransformer(config)
    x = torch.randint(0, config.vocab_size, (1, 4))
    y = model.generate(x, max_new_tokens=3, top_k=4)
    assert y.shape == (1, 7)

import torch

from models import SVFTransformer, SVFTransformerConfig, build_config_for_variant


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


def test_variant_builder_baseline_disables_svf_modules():
    base = SVFTransformerConfig(vocab_size=32, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    config = build_config_for_variant(base, "baseline")

    assert config.use_memory is False
    assert config.use_persistent_core is False
    assert config.use_structural_dynamics is False
    assert config.use_structural_loss is False


def test_variant_builder_full_svf_enables_all_modules():
    base = SVFTransformerConfig(vocab_size=32, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    config = build_config_for_variant(base, "svf")

    assert config.use_memory is True
    assert config.use_persistent_core is True
    assert config.use_structural_dynamics is True
    assert config.use_structural_loss is True


def test_variant_builder_core_dynamics_enables_dynamics_without_memory():
    base = SVFTransformerConfig(vocab_size=32, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    config = build_config_for_variant(base, "core_dynamics")

    assert config.use_memory is False
    assert config.use_persistent_core is True
    assert config.use_structural_dynamics is True
    assert config.use_structural_loss is False


def test_baseline_loss_matches_cross_entropy_only():
    base = SVFTransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
    )
    config = build_config_for_variant(base, "baseline")
    model = SVFTransformer(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))

    out = model(x, targets=y, use_memory=config.use_memory, write_memory=config.use_memory)

    assert out.loss is not None
    assert out.ce_loss is not None
    assert torch.allclose(out.loss, out.ce_loss)
    assert float(out.conservation_loss.item()) == 0.0
    assert float(out.drift_loss.item()) == 0.0


def test_memory_core_variant_uses_core_without_structural_regularization():
    base = SVFTransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
    )
    config = build_config_for_variant(base, "memory_core")
    model = SVFTransformer(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))

    out = model(x, targets=y, use_memory=config.use_memory, write_memory=config.use_memory)

    assert out.loss is not None
    assert out.ce_loss is not None
    assert config.use_persistent_core is True
    assert config.use_structural_loss is False
    assert torch.allclose(out.loss, out.ce_loss)

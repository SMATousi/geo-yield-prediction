import pytest
import torch
from torch import nn
import numpy as np

from models_multimodal_encoder import MultiModalEncoder
from util.norm_stats import get_norm_stats, normalize_modality


@pytest.mark.parametrize(
    ("kind", "cfg", "sample", "tokens"),
    [
        ("raster", {"in_channels": 1}, lambda: torch.randn(2, 1, 4, 6), 24),
        ("patch", {"in_channels": 1, "patch_size": 2, "num_patches": 4}, lambda: torch.randn(2, 1, 4, 4), 5),
        ("sar", {"in_channels": 2}, lambda: torch.randn(2, 3, 2, 4, 4), 16),
        ("timeseries", {"in_dim": 3}, lambda: torch.randn(2, 5, 3), 1),
        ("tabular", {"in_dim": 3}, lambda: torch.randn(2, 3), 1),
        ("categorical", {"num_classes": 8}, lambda: torch.randint(0, 8, (2, 4, 4)), 16),
        pytest.param(
            "grouped_vit",
            {"img_size": 4, "patch_size": 2, "in_chans": 2,
             "channel_groups": ((0,), (1,)), "depth": 1, "num_heads": 2,
             "channel_embed": 8},
            lambda: torch.randn(2, 2, 4, 4),
            1,
            marks=[
                pytest.mark.known_defect,
                pytest.mark.xfail(
                    strict=True,
                    reason="grouped_vit dispatch emits (B, D), not fusion-ready (B, L, D)",
                ),
            ],
        ),
    ],
)
def test_encoder_type_contract(kind, cfg, sample, tokens):
    encoder = MultiModalEncoder({"source": {"type": kind, **cfg}}, embed_dim=32)
    encoder.eval()
    with torch.no_grad():
        output = encoder({"source": sample()})["source"]
    assert output.shape == (2, tokens, 32)
    assert torch.isfinite(output).all()


def test_caller_supplied_module_is_dispatched():
    class FixedEncoder(nn.Module):
        def forward(self, x):
            return x.unsqueeze(1)

    encoder = MultiModalEncoder(
        {"source": {"type": "module", "module": FixedEncoder()}}, embed_dim=4
    )
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    torch.testing.assert_close(encoder({"source": value})["source"], value.unsqueeze(1))


@pytest.fixture
def mixed_encoder():
    model = MultiModalEncoder(
        {
            "dem": {"type": "raster", "in_channels": 1},
            "weather": {"type": "timeseries", "in_dim": 3, "norm_source": "unregistered"},
        },
        embed_dim=16,
    )
    model.eval()
    return model


@pytest.fixture
def mixed_inputs():
    return {
        "dem": torch.randn(2, 1, 4, 4),
        "weather": torch.randn(2, 5, 3),
    }


def test_forward_with_missing_all_present(mixed_encoder, mixed_inputs):
    with torch.no_grad():
        embeddings, mask = mixed_encoder.forward_with_missing(mixed_inputs)
    assert embeddings["dem"].shape == (2, 16, 16)
    assert embeddings["weather"].shape == (2, 1, 16)
    for name in mixed_inputs:
        assert mask[name].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("missing", ["dem", "weather"])
def test_forward_with_missing_whole_modality(mixed_encoder, mixed_inputs, missing):
    inputs = {name: value for name, value in mixed_inputs.items() if name != missing}
    with torch.no_grad():
        embeddings, mask = mixed_encoder.forward_with_missing(inputs)
    assert embeddings[missing].shape == (2, 1, 16)
    assert mask[missing].tolist() == [0.0, 0.0]
    torch.testing.assert_close(
        embeddings[missing][:, 0], mixed_encoder.missing_tokens[missing].expand(2, -1)
    )


def test_forward_with_missing_all_absent_has_defined_batch_fallback(mixed_encoder):
    embeddings, mask = mixed_encoder.forward_with_missing({})
    for name in mixed_encoder.encoders:
        assert embeddings[name].shape == (1, 1, 16)
        assert mask[name].tolist() == [0.0]


@pytest.mark.parametrize("mixed", ["dem", "weather"])
def test_forward_with_missing_mixed_rows(mixed_encoder, mixed_inputs, mixed):
    available = {"dem": torch.ones(2), "weather": torch.ones(2)}
    available[mixed] = torch.tensor([1.0, 0.0])
    with torch.no_grad():
        embeddings, mask = mixed_encoder.forward_with_missing(
            mixed_inputs, available=available
        )
        present_only = mixed_encoder.encoders[mixed](mixed_inputs[mixed][:1])
    assert mask[mixed].tolist() == [1.0, 0.0]
    torch.testing.assert_close(embeddings[mixed][:1], present_only)
    expected_missing = mixed_encoder.missing_tokens[mixed].expand_as(embeddings[mixed][1])
    torch.testing.assert_close(embeddings[mixed][1], expected_missing)


def test_patch_position_is_registered_before_optimizer():
    encoder = MultiModalEncoder(
        {"dem": {"type": "patch", "in_channels": 1, "patch_size": 2, "num_patches": 4}}, embed_dim=32
    )
    assert "encoders.dem.pos_embedding" in dict(encoder.named_parameters())
    with pytest.raises(ValueError, match="expected 4 patches"):
        encoder({"dem": torch.randn(2, 1, 6, 6)})


def test_patch_encoder_requires_known_layout():
    with pytest.raises(ValueError, match="num_patches"):
        MultiModalEncoder({"dem": {"type": "patch", "in_channels": 1}}, embed_dim=32)


def test_tensor_normalization_reaches_tabular_encoder():
    model = MultiModalEncoder(
        {"source": {"type": "tabular", "in_dim": 2, "norm_source": "s1a"}},
        embed_dim=16,
    ).eval()
    mean, std = get_norm_stats("s1a")
    raw = torch.tensor(mean + std, dtype=torch.float32).expand(2, -1).clone()
    with torch.no_grad():
        actual = model({"source": raw})["source"]
        expected = model.encoders["source"](torch.ones_like(raw))
    torch.testing.assert_close(actual, expected)


def test_sar_normalization_uses_channel_axis_after_time():
    model = MultiModalEncoder(
        {"source": {"type": "sar", "in_channels": 2, "norm_source": "s1a"}},
        embed_dim=16,
    ).eval()
    mean, std = get_norm_stats("s1a")
    raw = torch.as_tensor(mean + std).reshape(1, 1, 2, 1, 1).expand(2, 3, 2, 2, 2).clone()
    with torch.no_grad():
        actual, _ = model.forward_with_missing({"source": raw})
        expected = model.encoders["source"](torch.ones_like(raw))
    torch.testing.assert_close(actual["source"], expected)
    np.testing.assert_allclose(
        normalize_modality("s1a", raw.numpy(), channel_axis=2),
        np.ones(raw.shape, dtype=np.float32),
        rtol=1e-6,
    )

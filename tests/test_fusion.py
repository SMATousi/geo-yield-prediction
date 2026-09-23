import pytest
import torch

from models_latent_fusion import LatentFusionTransformer, get_extraction_layers
from models_multimodal_encoder import MultiModalEncoder


@pytest.fixture
def fusion():
    model = LatentFusionTransformer(
        embed_dim=32,
        num_latents=16,
        depth=4,
        num_heads=2,
        modality_embed=8,
        modalities={
            "dem": {"spatial": 4, "temporal": 1},
            "weather": {"spatial": 1, "temporal": 3},
        },
    )
    model.eval()
    return model


def test_fusion_handles_different_token_layouts_and_multiscale(fusion):
    embeddings = {"dem": torch.randn(2, 4, 32), "weather": torch.randn(2, 3, 32)}
    with torch.no_grad():
        output = fusion(embeddings)
        multiscale = fusion.forward_multiscale(embeddings)
    assert output.shape == (2, 16, 32)
    assert len(multiscale) == 4
    assert all(level.shape == output.shape for level in multiscale)
    assert torch.isfinite(output).all()


def test_fusion_accepts_subset_and_gathered_tokens(fusion):
    dem = torch.randn(2, 4, 32)
    keep = torch.tensor([[0, 2], [1, 3]])
    with torch.no_grad():
        partial = fusion({"dem": dem})
        gathered = fusion({"dem": dem}, ids_keep={"dem": keep})
    assert partial.shape == gathered.shape == (2, 16, 32)
    assert not torch.equal(partial, gathered)


def test_masked_modality_cannot_influence_fusion(fusion):
    dem = torch.randn(2, 4, 32)
    weather = torch.randn(2, 3, 32)
    mask = {"dem": torch.ones(2), "weather": torch.zeros(2)}
    with torch.no_grad():
        original = fusion({"dem": dem, "weather": weather}, mask=mask)
        changed = fusion({"dem": dem, "weather": weather + 1000}, mask=mask)
    torch.testing.assert_close(original, changed, rtol=0, atol=0)


def test_fusion_rejects_declared_layout_mismatch(fusion):
    with pytest.raises(ValueError, match="token|layout"):
        fusion({"dem": torch.randn(2, 3, 32)})
    with pytest.raises(ValueError, match="token|layout"):
        fusion.forward_multiscale({"dem": torch.randn(2, 3, 32)})
    with pytest.raises(ValueError, match="token|layout"):
        fusion({"dem": torch.randn(2, 3, 32)}, ids_keep={"dem": torch.tensor([[0, 1], [1, 2]])})


def test_all_absent_row_uses_finite_missing_token_fallback(fusion):
    embeddings = {"dem": torch.randn(2, 4, 32), "weather": torch.randn(2, 3, 32)}
    mask = {"dem": torch.tensor([0, 1]), "weather": torch.tensor([0, 0])}
    with torch.no_grad():
        output = fusion(embeddings, mask=mask)
        scales = fusion.forward_multiscale(embeddings, mask=mask)
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(level).all() for level in scales)


def test_fusion_accepts_encoder_output_when_every_modality_is_absent():
    encoder = MultiModalEncoder(
        {"dem": {"type": "raster", "in_channels": 1}}, embed_dim=32,
    )
    fusion = LatentFusionTransformer(
        embed_dim=32, num_latents=4, depth=1, num_heads=2,
        modalities={"dem": {"spatial": 4, "temporal": 1}}, modality_embed=8,
    )
    embeddings, mask = encoder.forward_with_missing({})
    with torch.no_grad():
        output = fusion(embeddings, mask=mask)
    assert output.shape == (1, 4, 32)
    assert torch.isfinite(output).all()


def test_extraction_layer_contract():
    assert get_extraction_layers(4) == [0, 1, 2, 3]
    assert get_extraction_layers(12) == [2, 5, 8, 11]
    with pytest.raises(ValueError):
        get_extraction_layers(0)

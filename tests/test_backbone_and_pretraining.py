import pytest
import torch

from models_multi_unified import MultiUnifiedModel
from models_multimodal_pretrain import MultimodalSelfSupervisedPretrain
from models_neck import MultiFusionNeck


def test_neck_fuses_native_feature_pyramids_to_output_grid():
    neck = MultiFusionNeck(
        embed_dim=16,
        in_feature_key=("dem", "soil"),
        feature_size=(2, 2),
        out_size=(6, 10),
        in_fusion_key_list=({"dem": 4, "soil": 4}, {"dem": 2}),
    )
    inputs = {
        "dem": {
            "encoder_features": torch.randn(2, 16, 2, 2),
            "features_list": [torch.randn(2, 2, 8, 8), torch.randn(2, 4, 4, 4)],
        },
        "soil": {
            "encoder_features": torch.randn(2, 16, 1, 1),
            "features_list": [torch.randn(2, 2, 2, 2), torch.randn(2, 4, 1, 1)],
        },
    }
    output = neck(inputs)
    assert output.shape == (2, 16, 6, 10)
    assert torch.isfinite(output).all()


def test_pretraining_objectives_are_finite_and_trainable():
    model = MultimodalSelfSupervisedPretrain(
        encoders_cfg={
            "dem": {"type": "raster", "in_channels": 1},
            "weather": {"type": "timeseries", "in_dim": 3},
        },
        embed_dim=32,
        num_latents=4,
        depth=1,
        num_heads=2,
        modality_embed=8,
        modalities={
            "dem": {"spatial": 4, "temporal": 1},
            "weather": {"spatial": 1, "temporal": 1},
        },
        modality_mask_ratio=0,
        decoder_embed_dim=16,
    )
    inputs = {"dem": torch.randn(2, 1, 2, 2), "weather": torch.randn(2, 4, 3)}
    total, losses = model(inputs)
    assert set(losses) == {"spatial", "modality", "temporal", "cross_modal", "contrastive"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    torch.testing.assert_close(total, sum(losses.values()))
    total.backward()
    assert model.encoders.encoders["dem"].proj.weight.grad is not None
    model.set_encoder_trainable(False)
    assert all(not param.requires_grad for param in model.encoders.parameters())
    model.set_encoder_trainable(True)
    assert all(param.requires_grad for param in model.encoders.parameters())


def test_unified_container_builds_two_registered_heads():
    model = MultiUnifiedModel(
        encoders_cfg={},
        heads_cfg=[{"type": "dense_yield"}, {"type": "field_average"}],
        embed_dim=32,
    )
    latent = torch.randn(2, 32, 4, 4)
    dense = model.heads[0](latent)
    scalar = model.heads[1](latent)
    assert dense.shape == (2, 1, 4, 4)
    assert scalar.shape == (2, 1)


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="G2: unified container does not connect encoder outputs to a tensor head")
def test_unified_container_end_to_end_forward():
    model = MultiUnifiedModel(
        encoders_cfg={"dem": {"type": "raster", "in_channels": 1}},
        heads_cfg=[{"type": "dense_yield"}],
        embed_dim=32,
    )
    assert model({"dem": torch.randn(2, 1, 4, 4)})[0].shape == (2, 1, 4, 4)

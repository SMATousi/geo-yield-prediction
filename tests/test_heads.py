import numpy as np
import pytest
import torch

from models_heads import (
    DenseYieldDPTHead,
    DenseYieldFCNHead,
    DenseYieldFPNHead,
    MDNYieldHead,
    get_head_info,
    list_heads,
    masked_yield_loss,
)


def test_head_registry_has_all_yield_decoders():
    assert {"dense_yield", "dense_yield_fpn", "dense_yield_dpt", "mdn_yield"} <= set(list_heads())
    assert get_head_info("dense_yield")["cls"] is DenseYieldFCNHead
    with pytest.raises(KeyError):
        get_head_info("unknown")


@pytest.mark.parametrize("loss_name", ["l1", "mse"])
def test_fcn_head_map_and_loss(loss_name):
    head = DenseYieldFCNHead(embed_dim=32, loss=loss_name)
    features = torch.randn(2, 32, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    output, loss = head(features, targets=target, mode="loss")
    assert output.shape == target.shape
    assert torch.isfinite(loss)
    loss.backward()
    assert head.head[0].weight.grad is not None


def test_fcn_head_reassembly_and_stitching():
    head = DenseYieldFCNHead(embed_dim=16)
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    grid = head.reassemble(values, np.arange(4), 2, 2)
    assert grid.shape == (2, 2, 4)
    np.testing.assert_array_equal(grid[0, 0], values[0])
    patch = np.ones((2, 2, 1), dtype=np.float32)
    full, coverage = head.stitch(patch, 0, 0, 2, 2, 0, 0, 4, 4, 4, 4)
    assert full.shape == (4, 4, 1)
    assert coverage == pytest.approx(1.0)


def test_fpn_head_map_and_loss():
    head = DenseYieldFPNHead(channels=64, out_channels=16)
    head.eval()
    features = [
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 16, 4, 4),
        torch.randn(2, 32, 2, 2),
        torch.randn(2, 64, 1, 1),
    ]
    target = torch.randn(2, 1, 8, 8)
    output, loss = head(features, targets=target, mode="loss")
    assert output.shape == target.shape
    assert torch.isfinite(loss)


def test_dpt_head_resamples_to_target_grid_and_loss():
    head = DenseYieldDPTHead(embed_dim=32, features=16)
    head.eval()
    features = [torch.randn(2, 32, size, size) for size in (8, 4, 2, 1)]
    target = torch.randn(2, 1, 6, 10)
    output, loss = head(features, target_size=(6, 10), targets=target, mode="loss")
    assert output.shape == target.shape
    assert torch.isfinite(loss)


def test_mdn_head_distribution_and_loss():
    head = MDNYieldHead(input_dim=16, hidden_dims=(16,), n_gaussians=3)
    head.eval()
    pi, mu, sigma = head(torch.randn(2, 16))
    assert pi.shape == mu.shape == sigma.shape == (2, 3)
    torch.testing.assert_close(pi.sum(dim=1), torch.ones(2))
    assert (sigma > 0).all()
    assert torch.isfinite(head.compute_loss(pi, mu, sigma, torch.randn(2)))
    assert head.sample(pi, mu, sigma, n_samples=3).shape == (2, 3)


@pytest.mark.parametrize("loss_name,expected", [("l1", 3.0), ("mse", 10.0)])
def test_nodata_and_nonfinite_yield_do_not_affect_loss_or_gradient(loss_name, expected):
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = torch.tensor([[[[2.0, -9999.0], [float("nan"), 4.0]]]])
    loss = masked_yield_loss(logits, target, loss=loss_name)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert logits.grad[0, 0, 0, 1] == 0
    assert logits.grad[0, 0, 1, 0] == 0
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("head", [
    DenseYieldFCNHead(embed_dim=16, log_target=True),
    DenseYieldFPNHead(channels=64, out_channels=16, log_target=True),
    DenseYieldDPTHead(embed_dim=16, features=16, log_target=True),
])
def test_each_dense_head_excludes_nodata_before_log_transform(head):
    logits = torch.zeros(1, 1, 1, 2)
    target = torch.tensor([[[[2.0, -9999.0]]]])
    loss = head.compute_loss(logits, target)
    assert loss.item() == pytest.approx(np.log(2.0))
    with pytest.raises(ValueError, match="no valid pixels"):
        head.compute_loss(logits, torch.full_like(target, -9999.0))


def test_valid_mask_can_exclude_field_boundary_cells():
    logits = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[2.0, 100.0], [4.0, 100.0]]]])
    valid_mask = torch.tensor([[[True, False], [True, False]]])
    assert masked_yield_loss(logits, target, valid_mask=valid_mask).item() == pytest.approx(3)
    with pytest.raises(ValueError, match="valid_mask"):
        masked_yield_loss(logits, target, valid_mask=torch.ones(2, 2))

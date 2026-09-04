import torch
from torch import nn

from uav_ican_3d.vision.lora import LoRAConv2d, inject_lora, merge_lora


def test_zero_initialized_adapter_preserves_output_and_merge() -> None:
    torch.manual_seed(4)
    model = nn.Sequential(nn.Conv2d(3, 5, 3, padding=1), nn.ReLU(), nn.Conv2d(5, 2, 1))
    inputs = torch.randn(2, 3, 8, 8)
    expected = model(inputs).detach()
    stats = inject_lora(model, rank=2, alpha=4, dropout=0.0, target_pattern=r"^(0|2)$")
    assert stats.adapted_layers == 2
    assert stats.trainable_parameters > 0
    assert all(
        parameter.requires_grad == ("lora_a" in name or "lora_b" in name)
        for name, parameter in model.named_parameters()
    )
    assert torch.allclose(model(inputs), expected)
    assert merge_lora(model) == 2
    assert not any(isinstance(module, LoRAConv2d) for module in model.modules())
    assert torch.allclose(model(inputs), expected)


def test_nonzero_adapter_merge_is_numerically_equivalent() -> None:
    torch.manual_seed(8)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, stride=2, padding=1))
    inject_lora(model, rank=3, alpha=6, dropout=0.0, target_pattern=r"^0$")
    with torch.no_grad():
        model[0].lora_b.weight.normal_()
    inputs = torch.randn(1, 3, 9, 11)
    expected = model(inputs).detach()
    merge_lora(model)
    assert torch.allclose(model(inputs), expected, atol=1e-6, rtol=1e-5)

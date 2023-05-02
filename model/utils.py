import torch


def image_to_patch(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    assert len(x.shape) == 4
    B, C, H, W = x.shape
    return (
        x.reshape(B, C, H // patch_size, patch_size,
                  W // patch_size, patch_size)
        .permute(0, 2, 4, 1, 3, 5)
        .flatten(1, 2)
        .flatten(2, 4)
    )

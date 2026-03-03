import torch
import torch.nn.functional as F


def masked_psnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor | None = None,
    data_range: float = 255.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute PSNR from the mean MSE over a batch of frames.

    1) For each frame:
         - Compute per-channel MSE over all channels.
         - If a mask is provided, use only spatial locations where mask == True.
         - Average MSE over channels.
    2) Average the per-frame MSE values across the batch.
    3) Convert the mean MSE to PSNR:
           PSNR = 20 * log10(data_range / sqrt(mean_mse))
       Return +inf if mean_mse <= 1e-10.

    Note: All zero mask will lead to infinite PSNR.

    Args:
        preds:   torch.Tensor of shape (N, C, H, W)
        targets: torch.Tensor of shape (N, C, H, W)
        masks:   Optional boolean torch.Tensor of shape (N, 1, H, W), (N, H, W) or (H, W).
                 If (H, W), the same mask is applied to all frames.
        data_range: Maximum possible pixel value (e.g. 1.0 or 255.0).

    Returns:
        Scalar torch.Tensor (float32): PSNR computed from mean MSE over frames.
    """
    if preds.ndim != 4 or targets.ndim != 4:
        raise ValueError("Expected preds/targets shape (N, C, H, W)")
    if preds.shape != targets.shape:
        raise ValueError("preds and targets must have the same shape")

    N, C, H, W = preds.shape

    if masks is None:
        masks = torch.ones((N, 1, H, W), dtype=torch.bool, device=preds.device)

    targets = targets.to(preds.device)
    masks = masks.to(preds.device)

    # Normalize mask to (N, 1, H, W) boolean if provided
    if masks.ndim == 2:
        if masks.shape != (H, W):
            raise ValueError("2D mask must have shape (H, W)")
        masks = masks[None, ...].expand(N, H, W).unsqueeze(1)
    elif masks.ndim == 3:
        if masks.shape != (N, H, W):
            raise ValueError("3D mask must have shape (N, H, W)")
        masks = masks.unsqueeze(1)
    elif masks.ndim == 4:
        if masks.shape != (N, 1, H, W):
            raise ValueError("4D mask must have shape (N, 1, H, W)")
    else:
        raise ValueError("Mask must have shape (H, W), (N, H, W) or (N, 1, H, W)")

    if masks.dtype != torch.bool:
        masks = masks.to(torch.bool)

    # Per-element squared error (float32 for stability)
    mse = F.mse_loss(
        preds.to(torch.float32),
        targets.to(torch.float32),
        reduction="none",
    )  # (N, C, H, W)

    nb_valid = masks.sum(dim=(2, 3))  # (N,1) number of valid pixels per frame

    num = (mse * masks).sum(dim=(2, 3))  # (N,C)
    denom = nb_valid.clamp_min(1).to(num.dtype)  # (N,1)
    mse_per_channel = num / denom  # (N,C)
    mse_per_frame = mse_per_channel.mean(dim=1)  # (N,)

    mean_mse = mse_per_frame.mean()

    # PSNR from mean MSE
    if mean_mse <= eps:
        return torch.tensor(float("inf"), device=preds.device, dtype=torch.float32)

    dr = torch.tensor(float(data_range), device=preds.device, dtype=torch.float32)
    return 20.0 * torch.log10(dr) - 10.0 * torch.log10(mean_mse)


"""
if __name__ == "__main__":
    target = load_full_video_frames(Path("/mnt/central_storage/data_pool_raw/AgiBotWorld-Alpha/observations/367/652444/videos/hand_left_color.mp4"))
    pred = load_full_video_frames(Path("/raid/andrew.mitri/agibot_cache/render/367/652444/head__to__hand_left__moge.mp4"))
    h, w = target.shape[-2:]
    mask = load_mask_mkv_ffv1(path=Path("/raid/andrew.mitri/agibot_cache/render/367/652444/head__to__hand_left__mask__moge.mkv"), width=w, height=h)
    mask = torch.from_numpy(mask.copy()).unsqueeze(1)  # (N,1,H,W)
    mask = mask > 127.5  # bool
    
    assert mask.sum(dim=(2, 3)).squeeze(1).min().item() > 0, "All frames have zero valid pixels in mask"
    print("Torch tensor shape, dtype, min/max:",
      target.shape, target.dtype, float(target.min()), float(target.max()))
    print("Torch tensor shape, dtype, min/max:",
      pred.shape, pred.dtype, float(pred.min()), float(pred.max()))
    print("Mask torch shape, dtype, sum:", mask.shape, mask.dtype, int(mask.sum()))
    
    psnr_value = masked_psnr(preds=pred, targets=target, masks=None)
    print(f"Video PSNR (Masked): {psnr_value:.3f} dB")
"""

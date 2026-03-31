# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
Self-forcing DMD2 Distillation for HDMap-conditioned Closed Loop (CL) models.

Adapts SelfForcingModel for MultiViewConditionCausal / control_input_hdmap_bbox
conditioning, replacing the action-based KV-cache rollout with HDMap-based slicing
and the SIL-style init_kv_cache API.
"""

from contextlib import contextmanager
from typing import Optional

import attrs
import torch

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_transfer2._src.imaginaire.utils import log, misc
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.predict2.modules.denoiser_scaling import VelocityPassthroughWrapper
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import TextEncoder
from cosmos_transfer2._src.predict2.utils.kv_cache import KVCacheConfig, VideoSeqPos

try:
    from projects.cosmos.sil.causal_multiview.configs.causal_cosmos2.defaults.conditioner import (  # pyrefly: ignore  # missing-import
        MultiViewConditionCausal,
    )
except ImportError:
    MultiViewConditionCausal = None  # type: ignore[assignment, misc]
from cosmos_transfer2._src.interactive.configs.method_configs.config_cosmos2_interactive_base import IS_PREPROCESSED_KEY
from cosmos_transfer2._src.interactive.methods.distribution_matching.self_forcing import (
    SelfForcingModel,
    SelfForcingModelConfig,
)


@attrs.define(slots=False)
class SelfForcingCLModelConfig(SelfForcingModelConfig):
    """Config for the HDMap-conditioned Self-Forcing CL model.

    SIL counterpart: DMDSelfForcingModelHDMapConfig in
    projects/cosmos/sil/causal_multiview/self_forcing/self_forcing_dmd_hdmap.py.
    """

    # RF-space noise level used when committing frames to the KV cache
    # (diffusion-forcing style).  SIL uses integer timestep 128 out of 1000,
    # which maps to ~0.128 in RF [0,1] space.  Set to 0 for teacher-forcing
    # (clean commit).
    context_noise: float = 0.128


class SelfForcingCLModel(SelfForcingModel):
    """Self-Forcing DMD2 distillation model for HDMap-conditioned CL (causal) generation.

    SIL counterpart: DMDSelfForcingModelHDMap in
    projects/cosmos/sil/causal_multiview/self_forcing/self_forcing_dmd_hdmap.py,
    which extends DMDSelfForcingModel (self_forcing_dmd.py) with HDMap preprocessing.
    This class adapts that SIL model to the interactive SelfForcingModel API, replacing
    the action-based KV-cache rollout with HDMap-based per-frame conditioning.
    """

    @misc.timer("SelfForcingCLModel: build_model")
    def build_model(self):
        # Text encoder setup
        self.text_encoder = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self.text_encoder = TextEncoder(self.config.text_encoder_config)

        # Negative text prompt embedding
        self.neg_embed = (
            easy_io.load(self.config.neg_embed_path) if getattr(self.config, "neg_embed_path", "") else None
        )
        use_neg_prompt_str = getattr(self.config, "use_neg_prompt_str", False)
        neg_prompt_str = getattr(self.config, "neg_prompt_str", None)
        if use_neg_prompt_str and neg_prompt_str:
            assert self.text_encoder is not None, "text_encoder is required when use_neg_prompt_str is enabled"
            caption_key = getattr(self.config, "input_caption_key", "ai_caption")
            neg_data_batch = {caption_key: [neg_prompt_str]}
            neg_embed = self.text_encoder.compute_text_embeddings_online(neg_data_batch, caption_key)
            if isinstance(neg_embed, torch.Tensor) and neg_embed.ndim == 3:
                self.neg_embed = neg_embed[0]
            else:
                self.neg_embed = neg_embed
            log.info(
                "Computed negative prompt embedding with shape: "
                f"{self.neg_embed.shape} for neg_prompt_str: {neg_prompt_str}"
            )

        self.net = self.build_net(self.config.net)

        super().build_model()

        self.condition_postprocessor = (
            lazy_instantiate(self.config.condition_postprocessor)
            if getattr(self.config, "condition_postprocessor", None)
            else None
        )

        # Expand tokenizer mean/std buffers for long-video inference.
        self._expand_tokenizer_mean_std_buffers()
        # Expand RoPE sequence buffer for long-video inference.
        self._expand_rope_max_t()

    # ---- Overrides for streaming inference with arbitrary-length videos ----

    def _normalize_video_databatch_inplace(self, data_batch, input_key=None):
        """Override to remove the ``state_t`` length assertion.

        The base class asserts ``original_length == expected_length`` where
        ``expected_length`` is derived from ``self.config.state_t`` (the
        training-time clip length, e.g. 93 frames).  For streaming inference
        with long videos the input can be arbitrarily long, so the assertion
        is replaced with a warning.
        """
        input_key = self.input_data_key if input_key is None else input_key
        if input_key not in data_batch:
            return

        if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
            assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
        else:
            assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
            data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
            data_batch[IS_PREPROCESSED_KEY] = True

        expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
        original_length = data_batch[input_key].shape[2]
        if original_length != expected_length:
            log.info(
                f"Input video length ({original_length}) differs from state_t-derived "
                f"length ({expected_length}); streaming generation will handle this.",
            )

    def _build_streaming_condition(
        self,
        data_batch: dict,
        gt_latent: torch.Tensor,
        num_conditional_frames: int = 1,
    ) -> MultiViewConditionCausal:
        """Build condition directly for streaming inference, bypassing ``set_video_condition``.

        The SIL fast-infer path never calls ``set_video_condition`` (which
        asserts ``T % state_t == 0``).  Instead it constructs the condition
        tensors manually.  We do the same: encode text, build the mask, and
        assemble a ``MultiViewConditionCausal`` with the full-length latent.
        """
        B, C_lat, T_lat, H_lat, W_lat = gt_latent.shape

        # --- Text embeddings ---
        caption_key = getattr(self.config, "input_caption_key", self.input_caption_key)
        if "t5_text_embeddings" in data_batch:
            crossattn_emb = data_batch["t5_text_embeddings"]
        elif self.text_encoder is not None:
            crossattn_emb = self.text_encoder.compute_text_embeddings_online(data_batch, caption_key)
        else:
            raise RuntimeError("No text embeddings found and no text encoder available.")

        # --- Condition mask: first num_conditional_frames are conditioned ---
        mask = torch.zeros(B, 1, T_lat, H_lat, W_lat, device=gt_latent.device, dtype=gt_latent.dtype)
        mask[:, :, :num_conditional_frames] = 1.0

        # --- HDMap condition (encode only if still in pixel space) ---
        hdmap = data_batch.get("control_input_hdmap_bbox")
        if hdmap is not None and hdmap.shape[1] == 3:
            hdmap = self.encode(hdmap)

        return MultiViewConditionCausal(
            crossattn_emb=crossattn_emb,
            data_type=None,
            padding_mask=data_batch.get("padding_mask"),
            fps=data_batch.get("fps"),
            gt_frames=gt_latent,
            condition_video_input_mask_B_C_T_H_W=mask,
            state_t=T_lat,
            view_indices_B_T=None,
            ref_cam_view_idx_sample_position=None,
            control_input_hdmap_bbox=hdmap,
        )

    def _expand_tokenizer_mean_std_buffers(self, max_latent_frames: int = 500) -> None:
        """Expand the tokenizer's video_mean / video_std temporal buffers.

        The WAN tokenizer defaults to ``load_mean_std=False`` which sets
        ``video_mean=0, video_std=1`` (identity normalization) but only
        allocates 50 temporal positions.  The SIL fast-infer path allocates
        ``max_video_length=500``.  We match that here so long videos don't
        exceed the buffer.
        """
        tok_model = self.tokenizer.model
        for name, fill in [("video_mean", 0.0), ("video_std", 1.0)]:
            buf = getattr(tok_model, name)
            if buf.shape[2] < max_latent_frames:
                old_t = buf.shape[2]
                setattr(
                    tok_model,
                    name,
                    torch.full((1, 1, max_latent_frames, 1, 1), fill, device=buf.device, dtype=buf.dtype),
                )
                log.info(f"Expanded tokenizer {name} from {old_t} to {max_latent_frames} latent frames")

    def _expand_rope_max_t(self, max_latent_frames: int = 500) -> None:
        """Expand the RoPE positional embedder's temporal sequence buffer.

        The ``VideoRopePosition3DEmb`` pre-allocates ``self.seq`` with length
        ``max(max_h, max_w, max_t)`` where ``max_t`` matches the training clip
        length.  During streaming inference, ``_forward_inference`` generates
        full RoPE from frame 0 to ``start_frame_for_rope + 1``, which exceeds
        ``max_t`` for long videos.  The SIL fast-infer path avoids this by
        using an offset-based RoPE, but the CL network uses the slice-based
        approach, so we simply expand the buffer.
        """
        pos_emb = getattr(self.net, "pos_embedder", None)
        if pos_emb is None:
            return
        old_max_t = getattr(pos_emb, "max_t", None)
        if old_max_t is None or old_max_t >= max_latent_frames:
            return
        new_len = max(pos_emb.max_h, pos_emb.max_w, max_latent_frames)
        old_len = pos_emb.seq.shape[0]
        if new_len > old_len:
            pos_emb.seq = torch.arange(new_len, dtype=torch.float, device=pos_emb.seq.device)
        pos_emb.max_t = max_latent_frames
        log.info(f"Expanded RoPE max_t from {old_max_t} to {max_latent_frames}, seq length from {old_len} to {new_len}")
        pos_emb.reset_parameters()

    # ---- Helpers ----

    def _slice_condition_frame(self, condition: MultiViewConditionCausal, t_idx: int) -> MultiViewConditionCausal:
        """Slice full-sequence condition to a single frame at index t_idx."""
        condition_dict = condition.to_dict()
        condition_dict.update(
            gt_frames=condition.gt_frames[:, :, t_idx : t_idx + 1],
            condition_video_input_mask_B_C_T_H_W=condition.condition_video_input_mask_B_C_T_H_W[
                :, :, t_idx : t_idx + 1
            ],
            control_input_hdmap_bbox=condition.control_input_hdmap_bbox[:, :, t_idx : t_idx + 1],
        )
        return MultiViewConditionCausal(**condition_dict)

    @contextmanager
    def _velocity_scaling(self):
        """Temporarily swap scaling_from_time to VelocityPassthroughWrapper.

        This makes denoise_edm_seq a plain forward pass (c_in=1, c_skip=0,
        c_out=1, c_noise=time) so the output is the raw velocity prediction.
        Used for SIL-pretrained networks that were not trained with EDM preconditioning.
        """
        saved = self.scaling_from_time
        self.scaling_from_time = VelocityPassthroughWrapper(self.sigma_data)
        try:
            yield
        finally:
            self.scaling_from_time = saved

    # ---- denoise_edm_seq override with SIL-style GT frame replacement ----

    def denoise_edm_seq(self, x_B_C_T_H_W, timesteps_B_T, condition, video_pos, kv_cache_cfg=None):
        """Override that adds SIL-style GT frame replacement before the parent call.

        When self._gt_frame_replace is True (set during generate_streaming_video),
        conditional frame pixels are replaced with GT frames and the
        sigma_conditional time override is skipped — the parent's EDM scaling
        (which is VelocityPassthroughWrapper during inference) then acts on the
        already-correct input.  When the flag is False the parent's default
        sigma_conditional handling is used unchanged.
        """
        if getattr(self, "_gt_frame_replace", False):
            # SIL-style: replace conditional frame pixels with GT, keep timestep as-is.
            if hasattr(condition, "gt_frames") and condition.gt_frames is not None:
                B, C, _, H, W = x_B_C_T_H_W.shape
                gt_frames = condition.gt_frames.type_as(x_B_C_T_H_W)
                mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(x_B_C_T_H_W)
                x_B_C_T_H_W = gt_frames * mask + x_B_C_T_H_W * (1 - mask)

            # Null out the condition mask so the parent's sigma_conditional override
            # sees all-zero mask and becomes a no-op (time stays unchanged).
            condition_dict = condition.to_dict()
            condition_dict["condition_video_input_mask_B_C_T_H_W"] = torch.zeros_like(
                condition.condition_video_input_mask_B_C_T_H_W
            )
            condition = type(condition)(**condition_dict)

        return super().denoise_edm_seq(x_B_C_T_H_W, timesteps_B_T, condition, video_pos, kv_cache_cfg)

    # ---- Frame-level generation (RF Euler, called by generate_streaming_video) ----

    @torch.no_grad()
    def generate_next_frame(
        self,
        condition: MultiViewConditionCausal,
        frame_noise: torch.Tensor,
        t_idx: int,
        start_idx: int,
        *,
        full_video_pos: VideoSeqPos,
        n_steps: int,
        enable_grad_on_last_hop: bool = False,
    ) -> torch.Tensor:
        """Generate the next latent video frame using rectified-flow Euler denoising.

        Uses self._rf_pairs (precomputed by generate_streaming_video) for the
        RF timestep schedule.  Called from both the autoregressive generation
        loop and the warmup phase (via _denoise_and_commit_frame).
        """
        B = frame_noise.shape[0]
        condition_frame = self._slice_condition_frame(condition, t_idx)
        cur_video_pos = full_video_pos.frame(t_idx)
        rf_pairs = self._rf_pairs

        latent = frame_noise
        for step_idx, (t_cur_rf, t_next_rf) in enumerate(rf_pairs):
            t_net = torch.full((B, 1), t_cur_rf, device=frame_noise.device, dtype=frame_noise.dtype)
            kv_cfg = KVCacheConfig(run_with_kv=True, store_kv=False, current_idx=t_idx)

            is_last_step = step_idx == len(rf_pairs) - 1
            grad_ctx = torch.enable_grad if (enable_grad_on_last_hop and is_last_step) else torch.no_grad
            with grad_ctx():
                velocity = self.denoise_edm_seq(
                    latent,
                    t_net,
                    condition_frame,
                    cur_video_pos,
                    kv_cfg,
                )
                # Euler step: x_{t_next} = x_t + (t_next - t_cur) * v
                latent = latent + (t_next_rf - t_cur_rf) * velocity

        return latent

    # ---- Streaming video generation (SIL-style, used during inference) ----

    def generate_streaming_video(
        self,
        condition: MultiViewConditionCausal,
        init_noise: torch.Tensor,
        n_steps: int,
        cache_frame_size: int = -1,
        enable_grad_on_last_hop: bool = False,
        use_cuda_graphs: bool = False,
        shift: float = 5.0,
        seed: int = 1,
    ) -> torch.Tensor:
        """Autoregressively generate a full latent video using SIL-style denoising.

        Uses rectified-flow Euler steps with raw velocity predictions via
        denoise_edm_seq + VelocityPassthroughWrapper (no EDM preconditioning).
        Timesteps are computed via get_rectified_flow_sampling_timesteps (same
        as the parent's teacher sampling path).
        """
        init_noise = init_noise.to(**self.tensor_kwargs)
        B, C, T, H, W = init_noise.shape
        start_idx = 1

        initial_latent = condition.gt_frames[:, :, :start_idx].clone()

        token_h = H // self.net.patch_spatial
        token_w = W // self.net.patch_spatial
        max_cache_size = T if cache_frame_size == -1 else cache_frame_size

        self.net.init_kv_cache(
            max_cache_size=max_cache_size,
            batch_size=B,
            device=init_noise.device,
            dtype=init_noise.dtype,
            token_h=token_h,
            token_w=token_w,
        )
        full_video_pos = VideoSeqPos(T=T, H=token_h, W=token_w)

        # Compute RF timesteps using the parent's existing helper (same schedule as
        # FlowUniPCMultistepScheduler.set_timesteps with the same shift).
        t_steps_rf = self.get_rectified_flow_sampling_timesteps(n_steps, shift)
        # e.g. [1.0, 0.9375, 0.833, 0.625, 0.0] for n_steps=4, shift=5.0

        # Use velocity (identity) scaling so denoise_edm_seq becomes a plain
        # forward pass, and enable GT frame replacement for conditional frames.
        self._gt_frame_replace = True
        self._rf_pairs = list(zip(t_steps_rf[:-1], t_steps_rf[1:]))
        frame_outputs = []

        with self._velocity_scaling():
            # Warmup: prefill KV cache with the GT conditional frame(s).
            # Only the commit is needed — the denoised output is not used.
            for f in range(start_idx):
                condition_frame = self._slice_condition_frame(condition, f)
                cur_video_pos = full_video_pos.frame(f)
                self._commit_kv_cache(
                    condition.gt_frames[:, :, f : f + 1].clone(),
                    condition_frame,
                    cur_video_pos,
                    f,
                    seed,
                )

            # Autoregressive generation loop (matches grandparent pattern).
            for t_idx in range(start_idx, T):
                frame_noise = init_noise[:, :, t_idx : t_idx + 1]
                latent = self.generate_next_frame(
                    condition,
                    frame_noise,
                    t_idx,
                    start_idx,
                    full_video_pos=full_video_pos,
                    n_steps=n_steps,
                    enable_grad_on_last_hop=enable_grad_on_last_hop,
                )
                frame_outputs.append(latent)

                # Commit: update KV cache with noised frame (diffusion forcing).
                condition_frame = self._slice_condition_frame(condition, t_idx)
                cur_video_pos = full_video_pos.frame(t_idx)
                self._commit_kv_cache(
                    latent,
                    condition_frame,
                    cur_video_pos,
                    t_idx,
                    seed,
                )

        self._gt_frame_replace = False
        self._rf_pairs = None

        # Use torch.cat instead of slice assignment to preserve gradient tracking
        # when enable_grad_on_last_hop=True (training path).
        output_latents = torch.cat(
            [initial_latent.to(dtype=init_noise.dtype)] + frame_outputs,
            dim=2,
        )
        return output_latents

    # ---- Private helpers ----

    def _commit_kv_cache(
        self,
        denoised: torch.Tensor,
        condition_frame: MultiViewConditionCausal,
        video_pos: VideoSeqPos,
        frame_idx: int,
        seed: int,
    ) -> None:
        """Run a forward pass with a (optionally noised) frame to update the KV cache.

        Uses diffusion-forcing style commits matching SIL's inference behaviour.
        The noise level is controlled by ``self.config.context_noise`` (RF space,
        default 0.128 ≈ SIL's integer timestep 128/1000).  Set to 0 for clean
        (teacher-forcing) commits.
        """
        B = denoised.shape[0]
        t_commit = self.config.context_noise
        t_net = torch.full((B, 1), t_commit, device=denoised.device, dtype=denoised.dtype)

        if t_commit < 1e-5:
            commit_input = denoised
        else:
            g = torch.Generator(device=denoised.device)
            g.manual_seed(seed + frame_idx * 42)
            noise = torch.randn(denoised.shape, dtype=torch.float32, device=denoised.device, generator=g)
            commit_input = (1.0 - t_commit) * denoised + t_commit * noise

        kv_cfg = KVCacheConfig(run_with_kv=True, store_kv=True, current_idx=frame_idx)
        with torch.no_grad():
            _ = self.denoise_edm_seq(commit_input, t_net, condition_frame, video_pos, kv_cfg)

    # ---- Top-level sample generation ----

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict,
        seed: int = 1,
        state_shape=None,
        n_sample=None,
        num_steps: int = 4,
        init_noise: Optional[torch.Tensor] = None,
        net_type: str = "student",
        guidance: float = 0.0,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples for the causal CL model.

        For the student net: uses frame-by-frame autoregressive KV-cache generation
        via generate_streaming_video with SIL-style rectified-flow Euler denoising.
        For the teacher net: falls back to the parent class's bidirectional generation.
        """
        if net_type == "teacher":
            return super().generate_samples_from_batch(
                data_batch,
                seed=seed,
                state_shape=state_shape,
                n_sample=n_sample,
                num_steps=num_steps,
                init_noise=init_noise,
                net_type=net_type,
            )

        # Student path: causal AR generation frame-by-frame with KV-cache.
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )

        # Encode the video and build condition directly (like SIL fast-infer),
        # bypassing get_data_and_condition / set_video_condition which assert
        # T % state_t == 0.
        raw_state = data_batch[input_key]
        gt_latent = self.encode(raw_state).contiguous().float()

        num_cf = data_batch.get("num_conditional_frames", 1)
        if isinstance(num_cf, torch.Tensor):
            num_cf = int(num_cf.item())
        condition = self._build_streaming_condition(data_batch, gt_latent, num_conditional_frames=num_cf)

        output_latents = self.generate_streaming_video(
            condition=condition,
            init_noise=init_noise,
            n_steps=num_steps,
            cache_frame_size=self.config.cache_frame_size,
            shift=shift,
            seed=seed,
        )

        return output_latents

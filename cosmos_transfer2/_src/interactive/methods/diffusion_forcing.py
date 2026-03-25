# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
Diffusion Forcing training for HDMap-conditioned Closed Loop (CL) models.

Trains a causal model from a bidirectional HDMap teacher using per-frame
independent noise levels and block-causal attention.  This is the intermediate
stage between the bidirectional teacher and Self-Forcing DMD2 distillation.

SIL counterpart: CausalT2VWan2pt1Model.training_step in
cosmos_transfer2/_src/av/causal/causal_training/models/t2v_model_causal.py.
"""

from typing import Any, Dict, Mapping

import attrs
import torch
from einops import rearrange

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_transfer2._src.imaginaire.modules.res_sampler import Sampler
from cosmos_transfer2._src.imaginaire.utils import log, misc
from cosmos_transfer2._src.imaginaire.utils.count_params import count_params
from cosmos_transfer2._src.interactive.methods.distribution_matching.self_forcing_cl import (
    SelfForcingCLModel,
    SelfForcingCLModelConfig,
)
from cosmos_transfer2._src.predict2.modules.denoiser_scaling import VelocityPassthroughWrapper
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import TextEncoder
from cosmos_transfer2._src.predict2.tokenizers.base_vae import BaseVAE
from cosmos_transfer2._src.predict2.utils.kv_cache import VideoSeqPos


@attrs.define(slots=False)
class DiffusionForcingCLModelConfig(SelfForcingCLModelConfig):
    """Config for the HDMap-conditioned Diffusion Forcing CL model.

    Inherits from SelfForcingCLModelConfig for inference compatibility
    (context_noise, cache_frame_size, etc.).  Distillation-specific fields
    (loss_scale_sid, net_teacher, etc.) are unused during training.
    """

    pass


class DiffusionForcingCLModel(SelfForcingCLModel):
    """Diffusion Forcing training model for HDMap-conditioned CL (causal) generation.

    Trains a single network with per-frame independent noise levels and
    block-causal attention.  Inherits KV-cache streaming inference from
    SelfForcingCLModel.

    Training:
        - Flow-matching velocity prediction with per-frame logit-normal time sampling
        - Block-causal attention mask (chunk_size=1, each frame attends to past + self)
        - Loss: MSE on velocity, masked to exclude conditional frames

    Inference:
        - KV-cache frame-by-frame streaming generation (inherited from SelfForcingCLModel)
    """

    # ---- Model construction ----

    @misc.timer("DiffusionForcingCLModel: build_model")
    def build_model(self):
        config = self.config

        # Text encoder
        self.text_encoder = None
        if config.text_encoder_config is not None and config.text_encoder_config.compute_online:
            self.text_encoder = TextEncoder(config.text_encoder_config)

        # Negative text prompt embedding
        self.neg_embed = None
        use_neg_prompt_str = getattr(config, "use_neg_prompt_str", False)
        neg_prompt_str = getattr(config, "neg_prompt_str", None)
        if use_neg_prompt_str and neg_prompt_str:
            assert self.text_encoder is not None, "text_encoder is required when use_neg_prompt_str is enabled"
            caption_key = getattr(config, "input_caption_key", "ai_caption")
            neg_data_batch = {caption_key: [neg_prompt_str]}
            neg_embed = self.text_encoder.compute_text_embeddings_online(neg_data_batch, caption_key)
            if isinstance(neg_embed, torch.Tensor) and neg_embed.ndim == 3:
                self.neg_embed = neg_embed[0]
            else:
                self.neg_embed = neg_embed
            log.info(
                f"Computed negative prompt embedding with shape: "
                f"{self.neg_embed.shape} for neg_prompt_str: {neg_prompt_str}"
            )

        # Core components
        self.sampler = Sampler()
        self.conditioner = lazy_instantiate(config.conditioner)
        assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
            "conditioner should not have learnable parameters"
        )
        self.tokenizer: BaseVAE = lazy_instantiate(config.tokenizer)
        assert self.tokenizer.latent_ch == config.state_ch, (
            f"latent_ch {self.tokenizer.latent_ch} != state_ch {config.state_ch}"
        )

        # Network (with FSDP)
        self.net = self.build_net(config.net)
        self._param_count = count_params(self.net, verbose=False)

        # Condition postprocessor (HDMap VAE encoding)
        self.condition_postprocessor = (
            lazy_instantiate(config.condition_postprocessor)
            if getattr(config, "condition_postprocessor", None)
            else None
        )

        # Velocity passthrough scaling: denoise_edm_seq returns raw velocity
        self.scaling_from_time = VelocityPassthroughWrapper(config.sigma_data)

        # No distillation networks
        self.net_teacher = None
        self.net_fake_score = None
        self.net_discriminator_head = None
        self.denoiser_nets = {"student": self.net}

        # No EMA for now (can be added if needed)
        self.latest_backward_simulation_video = None

        torch.cuda.empty_cache()

    # ---- Checkpointing (skip fake_score/discriminator from DMD2) ----

    def state_dict(self) -> Dict[str, Any]:
        """Only save the student net (no fake_score or discriminator)."""
        return torch.nn.Module.state_dict(self)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Load only the student net weights."""
        return torch.nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)

    def model_dict(self) -> Dict[str, Any]:
        """Only the student net for checkpointing."""
        return {"net": self.net}

    # ---- Optimizer (single net, no critic) ----

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        """Single optimizer for self.net (no distillation critics)."""
        from cosmos_transfer2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
        from cosmos_transfer2._src.imaginaire.utils.optim_instantiate import get_base_scheduler

        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": net_optimizer}
        net_scheduler = get_base_scheduler(net_optimizer, self, scheduler_config)
        self.scheduler_dict = {"net": net_scheduler}
        return net_optimizer, net_scheduler

    def is_student_phase(self, iteration: int) -> bool:
        """Always student phase (no critic alternation)."""
        return True

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        return [self.optimizer_dict["net"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        return [self.scheduler_dict["net"]]

    # ---- Training step ----

    def _sample_logit_normal_time(self, batch_size: int, num_frames: int, device: torch.device) -> torch.Tensor:
        """Sample per-frame RF times in [0, 1] using logit-normal distribution.

        Matches SIL's rectified_flow.sample_train_time with logit-normal sampling.
        Returns shape [B, T].
        """
        p_mean = self.config.sde["p_mean"]
        p_std = self.config.sde["p_std"]
        u = torch.randn(batch_size, num_frames, device=device) * p_std + p_mean
        t_B_T = torch.sigmoid(u)
        # Clamp to avoid exact 0 or 1
        t_B_T = t_B_T.clamp(min=1e-5, max=1.0 - 1e-5)
        return t_B_T

    def single_train_step(
        self, data_batch: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Diffusion forcing training step with per-frame noise levels.

        Matches SIL's CausalT2VWan2pt1Model.training_step (diffusion_forcing branch):
        1. Sample per-frame RF times with logit-normal distribution
        2. Create noisy frames via flow-matching interpolation (ALL frames, including conditional)
        3. Forward pass through net directly with block-causal attention
        4. Compute velocity prediction MSE loss on all frames
        """
        self._update_train_stats(data_batch)

        # Get latent video and conditions
        _, x0_B_C_T_H_W, condition, _ = self.get_data_and_condition(data_batch)
        B, C, T, H, W = x0_B_C_T_H_W.shape

        # Sample per-frame RF times [B, T] with logit-normal distribution
        # All frames get independent noise (including conditional), matching SIL
        t_B_T = self._sample_logit_normal_time(B, T, x0_B_C_T_H_W.device)

        # Sample noise
        noise_B_C_T_H_W = torch.randn_like(x0_B_C_T_H_W)

        # Flow-matching interpolation: x_t = (1 - t) * x0 + t * noise
        # SIL convention: x_t = noise * sigma + clean * (1 - sigma), with sigma = t
        t_B_1_T_1_1 = rearrange(t_B_T, "b t -> b 1 t 1 1")
        x_t_B_C_T_H_W = (1.0 - t_B_1_T_1_1) * x0_B_C_T_H_W + t_B_1_T_1_1 * noise_B_C_T_H_W

        # Target velocity: v = noise - x0
        v_target_B_C_T_H_W = noise_B_C_T_H_W - x0_B_C_T_H_W

        # Forward pass: call the network directly (matching SIL's training step)
        # The network handles block-causal mask and condition mask concatenation internally
        v_pred_B_C_T_H_W = self.net.forward_seq(
            x_B_C_T_H_W=x_t_B_C_T_H_W.to(**self.tensor_kwargs),
            video_pos=VideoSeqPos(T=T, H=H // self.net.patch_spatial, W=W // self.net.patch_spatial),
            timesteps_B_T=t_B_T.to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()

        # Loss: MSE per frame, all frames contribute (matching SIL)
        per_frame_loss = ((v_pred_B_C_T_H_W - v_target_B_C_T_H_W) ** 2).mean(dim=[1, 3, 4])  # [B, T]
        loss = per_frame_loss.mean()

        return {"df_loss": loss.detach()}, loss

    # ---- Inference (always use streaming, no teacher net) ----

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict,
        seed: int = 1,
        state_shape=None,
        n_sample=None,
        num_steps: int = 50,
        init_noise=None,
        net_type: str = "student",
        guidance: float = 0.0,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using KV-cache streaming (no teacher available).

        Always uses the student (only) net regardless of net_type.
        """
        # Route all requests through the student streaming path
        return super().generate_samples_from_batch(
            data_batch,
            seed=seed,
            state_shape=state_shape,
            n_sample=n_sample,
            num_steps=num_steps,
            init_noise=init_noise,
            net_type="student",
            guidance=guidance,
            shift=shift,
            **kwargs,
        )
